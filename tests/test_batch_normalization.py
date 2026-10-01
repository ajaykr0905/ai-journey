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
    snapshot_batch_norm,
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


class PyTorchParityTests(unittest.TestCase):
    def _native_forward(self, layer: torch.nn.BatchNorm1d, inputs: torch.Tensor):
        flattened = inputs.reshape(-1, inputs.shape[-1])
        return layer(flattened).reshape_as(inputs)

    def test_training_outputs_gradients_and_running_state_match_pytorch(self) -> None:
        torch.manual_seed(27)
        scratch = ScratchBatchNorm(4, eps=3e-5, momentum=0.2).double()
        native = torch.nn.BatchNorm1d(4, eps=3e-5, momentum=0.2).double()
        with torch.no_grad():
            scale = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.double)
            shift = torch.tensor([-0.2, 0.1, 0.3, -0.4], dtype=torch.double)
            scratch.weight.copy_(scale)
            scratch.bias.copy_(shift)
            native.weight.copy_(scale)
            native.bias.copy_(shift)

        for _ in range(2):
            values = torch.randn(3, 5, 4, dtype=torch.double)
            scratch_inputs = values.clone().requires_grad_(True)
            native_inputs = values.clone().requires_grad_(True)
            scratch_outputs = scratch(scratch_inputs)
            native_outputs = self._native_forward(native, native_inputs)
            self.assertTrue(
                torch.allclose(scratch_outputs, native_outputs, atol=1e-12, rtol=1e-10)
            )
            scratch_outputs.square().sum().backward()
            native_outputs.square().sum().backward()
            self.assertTrue(
                torch.allclose(
                    scratch_inputs.grad,
                    native_inputs.grad,
                    atol=1e-11,
                    rtol=1e-9,
                )
            )
            self.assertTrue(
                torch.allclose(
                    scratch.weight.grad, native.weight.grad, atol=1e-11, rtol=1e-9
                )
            )
            self.assertTrue(
                torch.allclose(
                    scratch.bias.grad, native.bias.grad, atol=1e-11, rtol=1e-9
                )
            )
            scratch.zero_grad(set_to_none=True)
            native.zero_grad(set_to_none=True)

        self.assertTrue(
            torch.allclose(scratch.running_mean, native.running_mean, atol=1e-12)
        )
        self.assertTrue(
            torch.allclose(scratch.running_var, native.running_var, atol=1e-12)
        )
        self.assertEqual(
            int(scratch.num_batches_tracked), int(native.num_batches_tracked)
        )

    def test_eval_outputs_match_pytorch_running_statistics(self) -> None:
        torch.manual_seed(28)
        scratch = ScratchBatchNorm(3, momentum=0.4)
        native = torch.nn.BatchNorm1d(3, momentum=0.4)
        for _ in range(3):
            inputs = torch.randn(4, 2, 3)
            scratch(inputs)
            self._native_forward(native, inputs)
        scratch.eval()
        native.eval()
        evaluation = torch.randn(2, 7, 3)
        self.assertTrue(
            torch.allclose(
                scratch(evaluation),
                self._native_forward(native, evaluation),
                atol=1e-6,
            )
        )


class BatchNormStateSnapshotTests(unittest.TestCase):
    def test_snapshot_fingerprints_complete_persistent_state(self) -> None:
        layer = ScratchBatchNorm(3, eps=1e-4, momentum=0.25)
        initial = snapshot_batch_norm(layer)
        self.assertEqual(initial.num_features, 3)
        self.assertEqual(initial.running_mean, (0.0, 0.0, 0.0))
        self.assertEqual(initial.running_var, (1.0, 1.0, 1.0))
        self.assertEqual(initial.num_batches_tracked, 0)
        self.assertEqual(len(initial.fingerprint()), 64)
        self.assertEqual(initial.fingerprint(), initial.fingerprint())

        layer(torch.tensor([[1.0, 2.0, 3.0], [4.0, 8.0, 12.0]]))
        updated = snapshot_batch_norm(layer)
        self.assertNotEqual(initial.fingerprint(), updated.fingerprint())
        self.assertEqual(updated.num_batches_tracked, 1)
        self.assertEqual(updated.weight, (1.0, 1.0, 1.0))
        self.assertEqual(updated.bias, (0.0, 0.0, 0.0))

    def test_snapshot_handles_non_affine_layers_and_rejects_invalid_state(self) -> None:
        layer = ScratchBatchNorm(2, affine=False)
        snapshot = snapshot_batch_norm(layer)
        self.assertIsNone(snapshot.weight)
        self.assertIsNone(snapshot.bias)
        layer.running_mean[0] = float("nan")
        with self.assertRaisesRegex(BatchNormalizationError, "finite"):
            snapshot_batch_norm(layer)
        with self.assertRaisesRegex(TypeError, "ScratchBatchNorm"):
            snapshot_batch_norm(torch.nn.BatchNorm1d(2))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
