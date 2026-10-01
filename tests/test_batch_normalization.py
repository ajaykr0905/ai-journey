from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from ai_journey.batch_normalization import (
    BatchNormalizationError,
    ScratchBatchNorm,
)


class ScratchBatchNormTests(unittest.TestCase):
    def test_constructor_validates_controls(self) -> None:
        for features in (0, -1):
            with (
                self.subTest(features=features),
                self.assertRaisesRegex(BatchNormalizationError, "positive"),
            ):
                ScratchBatchNorm(features)
        with self.assertRaisesRegex(TypeError, "integer"):
            ScratchBatchNorm(True)
        for momentum in (0.0, 1.1, float("inf")):
            with (
                self.subTest(momentum=momentum),
                self.assertRaises(BatchNormalizationError),
            ):
                ScratchBatchNorm(3, momentum=momentum)
        with self.assertRaisesRegex(BatchNormalizationError, "eps"):
            ScratchBatchNorm(3, eps=0)

    def test_training_normalizes_last_dimension_and_updates_running_state(self) -> None:
        layer = ScratchBatchNorm(3, momentum=0.25, affine=False)
        inputs = torch.tensor(
            [
                [[1.0, 4.0, -1.0], [3.0, 8.0, 1.0]],
                [[5.0, 6.0, 3.0], [7.0, 2.0, 5.0]],
            ]
        )
        outputs = layer(inputs)
        self.assertTrue(
            torch.allclose(outputs.mean(dim=(0, 1)), torch.zeros(3), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                outputs.var(dim=(0, 1), unbiased=False), torch.ones(3), atol=1e-4
            )
        )
        expected_mean = inputs.mean(dim=(0, 1))
        expected_var = inputs.var(dim=(0, 1), unbiased=True)
        self.assertTrue(torch.allclose(layer.running_mean, expected_mean * 0.25))
        self.assertTrue(
            torch.allclose(layer.running_var, torch.ones(3).lerp(expected_var, 0.25))
        )
        self.assertEqual(int(layer.num_batches_tracked), 1)

    def test_eval_uses_running_state_without_mutating_it(self) -> None:
        layer = ScratchBatchNorm(2, momentum=1.0)
        layer(torch.tensor([[1.0, 3.0], [5.0, 7.0]]))
        mean_before = layer.running_mean.clone()
        variance_before = layer.running_var.clone()
        count_before = layer.num_batches_tracked.clone()
        layer.eval()
        inputs = torch.tensor([[9.0, 11.0]])
        outputs = layer(inputs)
        expected = (inputs - mean_before) * torch.rsqrt(variance_before + layer.eps)
        self.assertTrue(torch.allclose(outputs, expected))
        self.assertTrue(torch.equal(layer.running_mean, mean_before))
        self.assertTrue(torch.equal(layer.running_var, variance_before))
        self.assertTrue(torch.equal(layer.num_batches_tracked, count_before))

    def test_affine_parameters_receive_gradients(self) -> None:
        layer = ScratchBatchNorm(2)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 6.0]], requires_grad=True)
        outputs = layer(inputs)
        loss = outputs.square().sum() + outputs.sum()
        loss.backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)

    def test_input_contract_rejects_invalid_shapes_and_dtypes(self) -> None:
        layer = ScratchBatchNorm(3)
        with self.assertRaisesRegex(TypeError, "floating"):
            layer(torch.ones(2, 3, dtype=torch.long))
        with self.assertRaisesRegex(BatchNormalizationError, "two dimensions"):
            layer(torch.ones(3))
        with self.assertRaisesRegex(BatchNormalizationError, "feature dimension"):
            layer(torch.ones(2, 4))
        with self.assertRaisesRegex(BatchNormalizationError, "more than one"):
            layer(torch.ones(1, 3))

    def test_reset_running_stats_restores_initial_state(self) -> None:
        layer = ScratchBatchNorm(2, momentum=1.0)
        layer(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        layer.reset_running_stats()
        self.assertTrue(torch.equal(layer.running_mean, torch.zeros(2)))
        self.assertTrue(torch.equal(layer.running_var, torch.ones(2)))
        self.assertEqual(int(layer.num_batches_tracked), 0)


if __name__ == "__main__":
    unittest.main()
