from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.neural_net import (
    default_parameters,
    finite_difference_gradients,
    forward,
    full_backward,
    gradient_error,
    output_bias_gradient,
    output_weight_gradient,
    torch_autograd_gradients,
)


class NeuralNetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = np.array([[0.2, 0.7], [0.9, 0.1]], dtype=np.float64)
        self.targets = np.array([[1.0], [0.0]], dtype=np.float64)
        self.parameters = default_parameters()
        self.cache = forward(self.inputs, self.parameters)

    def test_forward_shapes_and_range(self) -> None:
        self.assertEqual(self.cache["a1"].shape, (2, 2))
        self.assertEqual(self.cache["a2"].shape, (2, 1))
        self.assertTrue(np.all((self.cache["a2"] > 0) & (self.cache["a2"] < 1)))

    def test_day_7_and_8_are_slices_of_full_backward(self) -> None:
        gradients = full_backward(self.cache, self.targets, self.parameters)
        np.testing.assert_allclose(
            output_bias_gradient(self.cache, self.targets), gradients["b2"]
        )
        np.testing.assert_allclose(
            output_weight_gradient(self.cache, self.targets), gradients["w2"]
        )

    def test_manual_gradients_match_finite_differences(self) -> None:
        analytic = full_backward(self.cache, self.targets, self.parameters)
        numerical = finite_difference_gradients(
            self.inputs, self.targets, self.parameters
        )
        self.assertLess(max(gradient_error(analytic, numerical).values()), 1e-6)

    def test_manual_gradients_match_torch_when_available(self) -> None:
        torch_gradients = torch_autograd_gradients(
            self.inputs, self.targets, self.parameters
        )
        if torch_gradients is None:
            self.skipTest("PyTorch is not installed")
        analytic = full_backward(self.cache, self.targets, self.parameters)
        self.assertLess(max(gradient_error(analytic, torch_gradients).values()), 1e-9)


if __name__ == "__main__":
    unittest.main()
