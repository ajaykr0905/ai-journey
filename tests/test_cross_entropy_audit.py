"""Exercise the independent gradient oracle and its failure gates."""

import unittest
from unittest.mock import patch

import numpy as np

from ai_journey.cross_entropy import manual_cross_entropy
from ai_journey.cross_entropy_audit import audit_cross_entropy


class GradientAuditTests(unittest.TestCase):
    def test_seeded_shapes_match_both_oracles(self):
        rng = np.random.default_rng(28)
        for rows, classes in [(1, 2), (3, 5), (8, 7)]:
            with self.subTest(rows=rows, classes=classes):
                report = audit_cross_entropy(
                    rng.normal(size=(rows, classes)), rng.integers(classes, size=rows)
                )
                self.assertTrue(report.passed, report)
                self.assertLess(report.autograd_error, 1e-15)

    def test_extreme_logits_are_verified(self):
        report = audit_cross_entropy(
            [[1000, -1000], [-1000, 1000]], [1, 1], tolerance=1e-7
        )
        self.assertTrue(report.passed, report)
        self.assertEqual(report.manual_loss, 1000)

    def test_negative_control_rejects_a_wrong_manual_derivative(self):
        trace = manual_cross_entropy([[1, 2, 3]], [0])
        trace.dlogits[:] = 0
        with patch(
            "ai_journey.cross_entropy_audit.manual_cross_entropy", return_value=trace
        ):
            report = audit_cross_entropy([[1, 2, 3]], [0])
        self.assertFalse(report.passed)
        self.assertGreater(report.autograd_error, 0.5)

    def test_strict_zero_tolerance_exposes_numerical_error(self):
        report = audit_cross_entropy([[0.1, -0.2, 0.3]], [1], tolerance=0)
        self.assertFalse(report.passed)

    def test_rejects_invalid_probe_controls(self):
        for controls in [
            {"epsilon": 0},
            {"epsilon": float("nan")},
            {"tolerance": -1},
            {"tolerance": float("inf")},
        ]:
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                audit_cross_entropy([[1, 2]], [0], **controls)

    def test_rejects_unrepresentable_probe(self):
        with self.assertRaisesRegex(ValueError, "cannot perturb"):
            audit_cross_entropy([[1e20, 1e20]], [0])

    def test_does_not_modify_inputs_or_numpy_rng(self):
        values = np.array([[1.0, 2.0]])
        original = values.copy()
        state = np.random.get_state()
        audit_cross_entropy(values, [0])
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(np.random.get_state()[1], state[1])


if __name__ == "__main__":
    unittest.main()
