from argparse import Namespace
import unittest

import torch
from torch import nn

from model.config import sc_ch_dict
from model.model import SingleLiteNetPlus, TwinLiteNetPlus, UpConvBlock, model_from_weights
from utils import build_optimizers, train


class ModelTest(unittest.TestCase):
    def test_modern_models_keep_output_resolution(self):
        image = torch.rand(1, 3, 32, 64)
        for config in sc_ch_dict:
            for model_cls in (TwinLiteNetPlus, SingleLiteNetPlus):
                with self.subTest(config=config, model=model_cls.__name__), torch.inference_mode():
                    model = model_cls(Namespace(config=config)).eval()
                    self.assertIsInstance(model.encoder.level1.act, nn.SiLU)
                    self.assertFalse(any(isinstance(m, nn.ConvTranspose2d) for m in model.modules()))
                    output = model(image)
                    heads = output if isinstance(output, tuple) else (output,)
                    self.assertEqual(len(heads), 2 if model_cls is TwinLiteNetPlus else 1)
                    for logits in heads:
                        self.assertEqual(logits.shape, (1, 2, 32, 64))
                        self.assertTrue(torch.isfinite(logits).all())

    def test_head_returns_signed_logits_without_normalization(self):
        head = UpConvBlock(8, 2, last=True)
        with torch.no_grad():
            classifier = head.up_conv[0]
            classifier.weight.zero_()
            classifier.bias.copy_(torch.tensor([-2.] * 4 + [3.] * 4))
            expected = torch.tensor([-2., 3.]).view(1, 2, 1, 1).expand(1, 2, 8, 16)
            torch.testing.assert_close(head(torch.zeros(1, 8, 4, 8)), expected)

    def test_weights_select_matching_architecture(self):
        args = Namespace(config='nano')
        image = torch.rand(1, 3, 32, 64)
        for model_cls in (TwinLiteNetPlus, SingleLiteNetPlus):
            for legacy in (False, True):
                with self.subTest(model=model_cls.__name__, legacy=legacy), torch.inference_mode():
                    original = model_cls(args, legacy=legacy).eval()
                    loaded = model_from_weights(model_cls, args, original.state_dict()).eval()
                    activation = nn.PReLU if legacy else nn.SiLU
                    self.assertIsInstance(loaded.encoder.level1.act, activation)
                    torch.testing.assert_close(loaded(image), original(image))

    def test_nano_updates_encoder_attention_and_heads_with_muon(self):
        torch.manual_seed(0)
        model = TwinLiteNetPlus(Namespace(config='nano'))
        hyp = dict(lr=0.01, momentum=0.9, weight_decay=0.0, eps=1e-8)
        optimizers, conv_weights = build_optimizers(model, hyp)
        kernel_by_view = {id(view): kernel for kernel, view in conv_weights}
        muon_params = optimizers['muon'].param_groups[0]['params']
        self.assertTrue(all(p.ndim == 2 for p in muon_params))
        self.assertEqual(
            {id(kernel_by_view.get(id(p), p)) for p in muon_params},
            {id(p) for p in model.parameters() if p.ndim >= 2},
        )
        self.assertEqual(
            {id(p) for p in optimizers['adamw'].param_groups[0]['params']},
            {id(p) for p in model.parameters() if p.ndim == 1},
        )
        weights = (
            model.encoder.level2[0].c1.weight,
            model.caam.proj_query.weight,
            model.out_da.up_conv[0].weight,
            model.out_ll.up_conv[0].weight,
        )
        before = [p.detach().clone() for p in weights]

        def loss(outputs, target):
            return None, None, sum((logits - target).square().mean() for logits in outputs)

        batch = (None, torch.randint(0, 256, (2, 3, 32, 64)), torch.ones(2, 2, 32, 64))
        train(Namespace(), [batch], model, loss, optimizers, conv_weights, 0,
              torch.amp.GradScaler('cpu', enabled=False))
        for previous, weight in zip(before, weights):
            self.assertTrue(torch.isfinite(weight).all())
            self.assertFalse(torch.equal(previous, weight))
