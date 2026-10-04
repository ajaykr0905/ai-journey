from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_journey.batchnorm_backward import (
    audit_batchnorm_backward,
    manual_batchnorm_backward,
)


class BatchNormBackwardTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(30)
        self.x = rng.normal(size=(2, 3, 4))
        self.gamma = np.array([0.0, -2.0, 0.5, 3.0])
        self.beta = rng.normal(size=4)
        self.dy = rng.normal(size=self.x.shape)

    def test_every_intermediate_matches_autograd_and_leaf_finite_differences(self):
        for x in (self.x, self.x.reshape(6, 4), self.x[:1, :2]):
            with self.subTest(shape=x.shape):
                audit = audit_batchnorm_backward(
                    x,
                    self.gamma,
                    self.beta,
                    self.dy.reshape(-1, 4)[: x.size // 4].reshape(x.shape),
                )
                self.assertTrue(audit["passed"], audit)
                self.assertEqual(
                    set(audit["gradient_errors"]),
                    {
                        "inputs",
                        "scale",
                        "bias",
                        "mean",
                        "centered",
                        "squared",
                        "variance",
                        "inv_std",
                        "normalized",
                        "output",
                    },
                )
                self.assertEqual(
                    set(audit["finite_difference_errors"]), {"inputs", "scale", "bias"}
                )

    def test_native_batchnorm_matches_arbitrary_vector_jacobian_product(self):
        trace = manual_batchnorm_backward(self.x, self.gamma, self.beta, self.dy)
        x = torch.tensor(self.x.reshape(-1, 4), dtype=torch.double, requires_grad=True)
        layer = torch.nn.BatchNorm1d(4).double()
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(self.gamma))
            layer.bias.copy_(torch.from_numpy(self.beta))
        output = layer(x)
        output.backward(torch.from_numpy(self.dy.reshape(-1, 4)))
        np.testing.assert_allclose(
            output.detach().numpy(), trace.values["output"].reshape(-1, 4), atol=1e-12
        )
        np.testing.assert_allclose(
            x.grad.numpy(), trace.gradients["inputs"].reshape(-1, 4), atol=1e-12
        )
        np.testing.assert_allclose(
            layer.weight.grad.numpy(), trace.gradients["scale"], atol=1e-12
        )
        np.testing.assert_allclose(
            layer.bias.grad.numpy(), trace.gradients["bias"], atol=1e-12
        )

    def test_audit_uses_normalized_cpu_seed_regardless_of_input_byte_order_or_default_device(
        self,
    ):
        big_endian = self.dy.astype(np.dtype(">f8"))
        self.assertTrue(
            audit_batchnorm_backward(self.x, self.gamma, self.beta, big_endian)[
                "passed"
            ]
        )
        with torch.device("meta"):
            self.assertTrue(
                audit_batchnorm_backward(self.x, self.gamma, self.beta, self.dy)[
                    "passed"
                ]
            )

    def test_constant_features_use_epsilon_and_do_not_drop_input_gradient(self):
        x = np.ones((3, 2))
        dy = np.array([[1.0, -2.0], [3.0, 4.0], [-1.0, 5.0]])
        trace = manual_batchnorm_backward(x, [2.0, -1.0], [0.0, 3.0], dy)
        expected = (dy - dy.mean(axis=0)) * [2.0, -1.0] / np.sqrt(1e-5)
        np.testing.assert_allclose(trace.gradients["inputs"], expected, atol=1e-10)
        np.testing.assert_array_equal(trace.gradients["scale"], [0.0, 0.0])
        self.assertTrue(
            audit_batchnorm_backward(
                x, [2.0, -1.0], [0.0, 3.0], dy, epsilon=1e-7, tolerance=1e-5
            )["passed"]
        )

    def test_backward_is_translation_invariant_and_does_not_mutate_inputs(self):
        original = [v.copy() for v in (self.x, self.gamma, self.beta, self.dy)]
        trace = manual_batchnorm_backward(self.x, self.gamma, self.beta, self.dy)
        np.testing.assert_allclose(
            trace.gradients["inputs"].sum(axis=(0, 1)), 0.0, atol=1e-12
        )
        np.testing.assert_array_equal(trace.gradients["inputs"][..., 0], 0.0)
        for actual, expected in zip((self.x, self.gamma, self.beta, self.dy), original):
            np.testing.assert_array_equal(actual, expected)

    def test_contract_rejects_nonfinite_wrong_shapes_and_nonreal_values(self):
        cases = [
            ([], [], [], []),
            ([[1.0, 2.0]], [1.0, 1.0], [0.0, 0.0], [[1.0, 1.0]]),
            (self.x, [1.0], self.beta, self.dy),
            (self.x, self.gamma, self.beta, [1.0]),
            (self.x * np.nan, self.gamma, self.beta, self.dy),
            (self.x, self.gamma.astype(complex), self.beta, self.dy),
            (self.x, self.gamma, self.beta, self.dy.astype(bool)),
        ]
        for case in cases:
            with self.subTest(case=str(case)[:50]), self.assertRaises(ValueError):
                manual_batchnorm_backward(*case)
        for eps in (True, 0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(eps=eps), self.assertRaises((TypeError, ValueError)):
                manual_batchnorm_backward(
                    self.x, self.gamma, self.beta, self.dy, eps=eps
                )

    def test_audit_rejects_invalid_controls_and_unrepresentable_perturbations(self):
        for name in ("epsilon", "tolerance"):
            for value in (True, 0.0, -1.0, float("nan")):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    audit_batchnorm_backward(
                        self.x, self.gamma, self.beta, self.dy, **{name: value}
                    )
        with self.assertRaisesRegex(ValueError, "cannot perturb"):
            audit_batchnorm_backward(
                self.x, self.gamma, self.beta, self.dy, epsilon=1e-30
            )

    def test_audit_fails_when_manual_input_gradient_is_wrong(self):
        def wrong_backward(*args, **kwargs):
            trace = manual_batchnorm_backward(*args, **kwargs)
            trace.gradients["inputs"].fill(0.0)
            return trace

        with patch(
            "ai_journey.batchnorm_backward.manual_batchnorm_backward", wrong_backward
        ):
            audit = audit_batchnorm_backward(self.x, self.gamma, self.beta, self.dy)
        self.assertFalse(audit["passed"])
        self.assertGreater(audit["gradient_errors"]["inputs"], audit["tolerance"])
        self.assertGreater(
            audit["finite_difference_errors"]["inputs"], audit["tolerance"]
        )

    def test_cli_retains_failed_gate_report_and_preserves_report_on_invalid_input(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts/check_batchnorm_backward.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            command = [sys.executable, str(script), "--output", str(output)]
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            original = output.read_bytes()
            repeated = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False
            )
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertEqual(output.read_bytes(), original)
            invalid = subprocess.run(
                command + ["--epsilon", "nan"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(invalid.returncode, 2)
            self.assertEqual(output.read_bytes(), original)
            failed = subprocess.run(
                command + ["--tolerance", "1e-30"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(failed.returncode, 1, failed.stderr)
            self.assertFalse(json.loads(output.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
