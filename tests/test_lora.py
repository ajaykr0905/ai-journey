import unittest

import torch
from torch import nn

from ai_journey.lora import LoRAConfig, LoRAError, LoRALinear


class LoRAConfigTests(unittest.TestCase):
    def test_capacity_and_scaling_are_predeclared(self):
        self.assertEqual(LoRAConfig(rank=4, alpha=8).scaling, 2)

    def test_invalid_controls_cannot_silently_change_the_experiment(self):
        for name, values in {
            "rank": (True, 0, 257, 1.5),
            "alpha": (True, 0, -1, float("nan"), float("inf"), 1025),
            "dropout": (True, -0.1, 1, float("nan")),
            "seed": (True, -1, 2**32, 1.5),
        }.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(LoRAError):
                    LoRAConfig(**{name: value})


class LoRALinearTests(unittest.TestCase):
    def test_zero_update_preserves_base_output_and_constructor_rng(self):
        base = nn.Linear(4, 6).double().eval()
        inputs = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        rng = torch.get_rng_state().clone()
        layer = LoRALinear(base, LoRAConfig())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(base(inputs), layer(inputs)))
        self.assertTrue(all(p.requires_grad for p in base.parameters()))
        self.assertTrue(all(not p.requires_grad for p in layer.base.parameters()))
        self.assertFalse(layer.training)
        self.assertNotEqual(base.weight.data_ptr(), layer.base.weight.data_ptr())

    def test_low_rank_update_matches_dense_matrix_and_input_gradients(self):
        base = nn.Linear(4, 6).double()
        layer = LoRALinear(base, LoRAConfig(rank=2, alpha=3))
        with torch.no_grad():
            layer.lora_b.copy_(torch.arange(12).reshape(6, 2) / 10)
        x = torch.arange(8, dtype=torch.float64).reshape(2, 4).requires_grad_()
        expected = nn.functional.linear(
            x, base.weight + layer.delta_weight(), base.bias
        )
        actual = layer(x)
        torch.testing.assert_close(actual, expected)
        expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
        actual.sum().backward()
        torch.testing.assert_close(x.grad, expected_gradient)
        self.assertTrue(all(p.grad is None for p in layer.base.parameters()))
        self.assertGreater(float(layer.lora_a.grad.abs().sum()), 0)
        self.assertGreater(float(layer.lora_b.grad.abs().sum()), 0)

    def test_rank_and_finite_base_are_checked_before_allocation(self):
        with self.assertRaises(LoRAError):
            LoRALinear(nn.Linear(1, 3), LoRAConfig(rank=2))
        base = nn.Linear(4, 4)
        with torch.no_grad():
            base.weight[0, 0] = float("nan")
        with self.assertRaises(LoRAError):
            LoRALinear(base, LoRAConfig())

    def test_dropout_obeys_train_eval_modes(self):
        layer = LoRALinear(nn.Linear(4, 4), LoRAConfig(dropout=0.5))
        with torch.no_grad():
            layer.lora_b.fill_(1)
        inputs = torch.ones(20, 4)
        layer.eval()
        self.assertTrue(torch.equal(layer(inputs), layer(inputs)))
        expected = nn.functional.linear(
            inputs, layer.weight + layer.delta_weight(), layer.bias
        )
        torch.testing.assert_close(layer(inputs), expected)
        layer.train()
        self.assertFalse(torch.equal(layer(inputs), layer(inputs)))
