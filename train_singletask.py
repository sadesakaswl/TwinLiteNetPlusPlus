import os
from functools import partial
import torch
import torch.optim.lr_scheduler
import torch.backends.cudnn as cudnn
import yaml
import math
from copy import deepcopy
from argparse import ArgumentParser, BooleanOptionalAction

from model.model import SingleLiteNetPlus
from loss import SigleLoss
from utils import build_optimizers, load_optimizer_state, train, val_one, netParams, save_checkpoint, poly_lr_scheduler
from utils import parallel_validation, validation_loader
import BDD100K

class ModelEMA:
    """Exponential Moving Average (EMA) for model parameters"""
    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # Exponential decay ramp
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Update EMA model parameters"""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

def validation_report(loader, model, args):
    results = val_one(loader, model, args=args)
    if args.task == "DA":
        return f"Driving Area Segment: mIOU({results[2]:.3f})"
    return f"Lane Line Segment: Acc({results[0]:.3f}) IOU({results[1]:.3f})"


def train_net(args, hyp):
    """Train the neural network model with given arguments and hyperparameters"""
    use_ema = args.ema
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    
    model = SingleLiteNetPlus(args)
    checkpoint = None
    if args.resume:
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
    # if num_gpus > 1:
    #     model = torch.nn.DataParallel(model)
    
    os.makedirs(args.savedir, exist_ok=True)  # Ensure save directory exists
    
    trainLoader = torch.utils.data.DataLoader(
        BDD100K.DatasetOneTask(hyp, valid=False,task=args.task),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=cuda_available,
        persistent_workers=args.num_workers > 0)
    
    if cuda_available:
        model = model.cuda()
        cudnn.benchmark = True
    
    print(f'Total network parameters: {netParams(model)}')
    
    criteria = SigleLoss(hyp,task=args.task)
    start_epoch = 0
    optimizers, conv_weights = build_optimizers(model, hyp)
    
    ema = ModelEMA(model) if use_ema else None
    
    # Resume training from checkpoint
    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        if use_ema:
            ema.ema.load_state_dict(checkpoint['ema_state_dict'])
            ema.updates = checkpoint['updates']
        load_optimizer_state(optimizers, checkpoint['optimizer'])
        print(f"=> Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    
    forward = torch.compile(model, mode='max-autotune') if getattr(args, 'compile', True) else None
    scaler = torch.amp.GradScaler('cuda', enabled=cuda_available)
    
    device = next(model.parameters()).device
    dataset_type = partial(BDD100K.DatasetOneTask, task=args.task)
    with parallel_validation(args, hyp, SingleLiteNetPlus, dataset_type, validation_report, device) as submit_validation:
        if submit_validation is None:
            valLoader = validation_loader(dataset_type(hyp, valid=True), args, cuda_available)

        for epoch in range(start_epoch, args.max_epochs):
            model_file_name = os.path.join(args.savedir, f'model_{epoch}.pth')
            lr = poly_lr_scheduler(args, hyp, optimizers, epoch)
            print(f"Learning rate: {lr}")

            ema = train(args, trainLoader, model, criteria, optimizers, conv_weights, epoch, scaler, args.verbose, ema,
                        forward=forward)
            if submit_validation is None:
                print(validation_report(valLoader, ema.ema if use_ema else model, args))

            torch.save((ema.ema if use_ema else model).state_dict(), model_file_name)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema.ema.state_dict() if use_ema else None,
                'updates': ema.updates if use_ema else None,
                'optimizer': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                'lr': lr
            }, os.path.join(args.savedir, 'checkpoint.pth.tar'))
            if submit_validation is not None:
                submit_validation(epoch, model_file_name)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100, help='Max number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--savedir', default='./testv3', help='Directory to save the results')
    parser.add_argument('--task', type=str, choices=["DA", "LL"], default=None, help="DA for drivable area, LL for lane line")
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='Path to hyperparameters YAML')
    parser.add_argument('--resume', type=str, default='', help='Resume training from a checkpoint')
    parser.add_argument('--config', default='nano', help='Model configuration')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--ema', action='store_true', help='Use Exponential Moving Average (EMA)')
    parser.add_argument('--parallel-val', action=BooleanOptionalAction, default=True,
                        help='Validate saved epoch weights in a background process (default: enabled)')
    parser.add_argument('--compile', action=BooleanOptionalAction, default=True,
                        help='Use torch.compile (default: enabled)')
    args = parser.parse_args()
    
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # Load hyperparameters
    
    train_net(args, hyp.copy())
