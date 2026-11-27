from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing import get_context

import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import torch.nn as nn
from const import *
import yaml
import matplotlib
import matplotlib.pyplot as plt


LOGGING_NAME="custom"
def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def build_optimizers(model, hyp):
    matrices, vectors, conv_weights = [], [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim > 2:
            # Share storage so Muon updates the kernel without changing its model shape.
            matrix = param.detach().view(param.shape[0], -1)
            conv_weights.append((param, matrix))
            matrices.append(matrix)
        elif param.ndim == 2:
            matrices.append(param)
        else:
            vectors.append(param)

    options = dict(lr=hyp['lr'], weight_decay=hyp['weight_decay'])
    return {
        'muon': torch.optim.Muon(matrices, adjust_lr_fn='match_rms_adamw', **options),
        'adamw': torch.optim.AdamW(vectors, betas=(hyp['momentum'], 0.999), eps=hyp['eps'], **options),
    }, conv_weights


def load_optimizer_state(optimizers, state):
    if 'param_groups' in state:
        print("=> Legacy AdamW checkpoint: keeping model weights and resetting optimizer state.")
        return
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(state[name])
    muon = optimizers['muon']
    for group in muon.param_groups:
        group['momentum'] = muon.defaults['momentum']


def poly_lr_scheduler(args, hyp, optimizers, epoch, power=1.5):
    lr = round(hyp['lr'] * (1 - epoch / args.max_epochs) ** power, 8)
    for optimizer in optimizers.values():
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def train(args, train_loader, model, criterion, optimizers, conv_weights, epoch, scaler, verbose=False, ema=None, forward=None):
    model.train()
    forward = model if forward is None else forward
    device = next(model.parameters()).device
    print("epoch: ", epoch)
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    if verbose:
        LOGGER.info(('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss'))
        pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_,input, target) in pbar:
        model.zero_grad(set_to_none=True)
        input = input.to(device, dtype=torch.float32, non_blocking=True) / 255.0
        output = forward(input)
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            focal_loss,tversky_loss,loss = criterion(output,target)

        scaler.scale(loss).backward()
        for param, matrix in conv_weights:
            matrix.grad = None if param.grad is None else param.grad.reshape_as(matrix)
        for optimizer in optimizers.values():
            scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        if verbose and (i == 0 or (i + 1) % 10 == 0 or i + 1 == total_batches):
            metrics = torch.stack((tversky_loss, focal_loss, loss.detach())).tolist()
            pbar.set_description(('%13s' * 1 + '%13.4g' * 3) %
                                     (f'{epoch}/{300 - 1}', *metrics))
    return ema if ema is not None else None




@torch.no_grad()
def val(val_loader = None, model = None, half = False, args=None):

    model.eval()
    device = next(model.parameters()).device

    DA=SegmentationMetric(2)
    LL=SegmentationMetric(2)

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()

    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    if args.verbose:
        pbar = tqdm(pbar, total=total_batches)
    for i, (_,input, target) in pbar:
        input = input.to(device, dtype=torch.float32) / 255.0
        with torch.autocast(device.type, enabled=half and device.type == 'cuda'):
            output = model(input)

        ###-------------Drivable Segmetation--------------
        out_da = output[0]
        target_da = target[0]

        _,da_predict = torch.max(out_da, 1)
        da_predict = da_predict[:,12:-12]
        _,da_gt=torch.max(target_da, 1)

        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())

        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()

        da_acc_seg.update(da_acc,input.size(0))
        da_IoU_seg.update(da_IoU,input.size(0))
        da_mIoU_seg.update(da_mIoU,input.size(0))
        ###-------------Drivable Segmetation--------------

        ###-------------Lane Segmetation-----------------
        out_ll = output[1]
        target_ll = target[1]


        _,ll_predict=torch.max(out_ll, 1)
        ll_predict = ll_predict[:,12:-12]
        _,ll_gt=torch.max(target_ll, 1)
        
        LL.reset()
        LL.addBatch(ll_predict.cpu(), ll_gt.cpu())

        ll_acc = LL.lineAccuracy()
        ll_IoU = LL.IntersectionOverUnion()
        ll_mIoU = LL.meanIntersectionOverUnion()

        ll_acc_seg.update(ll_acc,input.size(0))
        ll_IoU_seg.update(ll_IoU,input.size(0))
        ll_mIoU_seg.update(ll_mIoU,input.size(0))
        ###-------------Lane Segmetation-----------------

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)
    
    return da_segment_result,ll_segment_result


@torch.no_grad()
def val_one(val_loader = None, model = None, half = False, args=None):

    model.eval()
    device = next(model.parameters()).device

    RE=SegmentationMetric(2)

    acc_seg = AverageMeter()
    IoU_seg = AverageMeter()
    mIoU_seg = AverageMeter()

    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    if args.verbose:
        pbar = tqdm(pbar, total=total_batches)
    for i, (_,input, target) in pbar:
        input = input.to(device, dtype=torch.float32) / 255.0
        with torch.autocast(device.type, enabled=half and device.type == 'cuda'):
            output = model(input)




        _,predict=torch.max(output, 1)
        predict = predict[:,12:-12]
        _,gt=torch.max(target, 1)
        
        RE.reset()
        RE.addBatch(predict.cpu(), gt.cpu())

        acc = RE.lineAccuracy()
        IoU = RE.IntersectionOverUnion()
        mIoU = RE.meanIntersectionOverUnion()

        acc_seg.update(acc,input.size(0))
        IoU_seg.update(IoU,input.size(0))
        mIoU_seg.update(mIoU,input.size(0))


    segment_result = (acc_seg.avg,IoU_seg.avg,mIoU_seg.avg)
    
    return segment_result


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def validation_loader(dataset, args, pin_memory):
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory, persistent_workers=args.num_workers > 0)


_validation_context = None


def _init_validation(args, hyp, model_type, dataset_type, report, device):
    global _validation_context
    model = model_type(args).to(device)
    loader = validation_loader(dataset_type(hyp, valid=True), args, device.type == 'cuda')
    _validation_context = model, loader, report, args


def _validate_checkpoint(epoch, weights_path):
    model, loader, report, args = _validation_context
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    metrics = report(loader, model, args).replace('\n', '; ')
    print(f"Validation epoch {epoch + 1}: {metrics}", flush=True)


@contextmanager
def parallel_validation(args, hyp, model_type, dataset_type, report, device):
    if not getattr(args, 'parallel_val', True):
        yield None
        return

    pending = deque()
    with ProcessPoolExecutor(
        max_workers=1, mp_context=get_context('spawn'), initializer=_init_validation,
        initargs=(args, hyp, model_type, dataset_type, report, device),
    ) as executor:
        def submit(epoch, weights_path):
            while pending and pending[0].done():
                pending.popleft().result()
            pending.append(executor.submit(_validate_checkpoint, epoch, weights_path))

        try:
            yield submit
            while pending:
                pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()


def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])
