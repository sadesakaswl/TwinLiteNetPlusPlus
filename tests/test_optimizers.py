from argparse import Namespace
from copy import deepcopy
import unittest

import torch
from torch import nn

from utils import build_optimizers, load_optimizer_state, poly_lr_scheduler, train


class MuonTrainingTest(unittest.TestCase):
    def test_updates_kernels_and_vectors_and_restores_both_states(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(2, 3, 3), nn.Flatten(), nn.Linear(12, 2))
        hyp = {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0, 'eps': 1e-8}
        args = Namespace(max_epochs=4)
        optimizers, conv_weights = build_optimizers(model, hyp)
        before = [p.detach().clone() for p in model.parameters()]
        batch = (None, torch.randint(0, 256, (2, 2, 4, 4)), torch.zeros(2, 2))

        def loss(output, target):
            return None, None, nn.functional.mse_loss(output, target)

        self.assertEqual(poly_lr_scheduler(args, hyp, optimizers, 2), 0.00353553)
        for optimizer in optimizers.values():
            self.assertEqual(optimizer.param_groups[0]['lr'], 0.00353553)
        train(args, [batch], model, loss, optimizers, conv_weights, 2, torch.amp.GradScaler('cpu', enabled=False))
        for param, initial in zip(model.parameters(), before):
            self.assertFalse(torch.equal(param, initial))
        self.assertEqual([p.shape for p in optimizers['muon'].state], [(3, 18), (2, 12)])
        self.assertEqual(set(optimizers['adamw'].state), {model[0].bias, model[2].bias})

        saved = {name: optimizer.state_dict() for name, optimizer in optimizers.items()}
        fresh_model = deepcopy(model)
        fresh, _ = build_optimizers(fresh_model, hyp)
        legacy = torch.optim.AdamW(model.parameters()).state_dict()
        load_optimizer_state(fresh, legacy)
        self.assertTrue(all(not optimizer.state for optimizer in fresh.values()))
        load_optimizer_state(fresh, saved)
        for name, optimizer in fresh.items():
            reloaded = optimizer.state_dict()
            self.assertEqual(saved[name]['param_groups'], reloaded['param_groups'])
            for index, state in saved[name]['state'].items():
                for key, tensor in state.items():
                    torch.testing.assert_close(reloaded['state'][index][key], tensor)
