"""Verify the model-wide loss derivative comparison and RNG isolation."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from ai_journey.cross_entropy import manual_cross_entropy
from ai_journey.transformer_gradient_audit import audit_transformer_loss_gradient


class TransformerGradientAuditTests(unittest.TestCase):
    def test_all_parameter_gradients_match(self):
        report = audit_transformer_loss_gradient()
        self.assertTrue(report.passed, report)
        self.assertGreater(len(report.parameter_errors), 10)
        self.assertIn("token_embedding.weight", report.parameter_errors)
        self.assertIn("lm_head.weight", report.parameter_errors)
        self.assertLess(max(report.parameter_errors.values()), 1e-12)

    def test_repeated_runs_are_identical(self):
        self.assertEqual(
            audit_transformer_loss_gradient(), audit_transformer_loss_gradient()
        )

    def test_uses_cpu_even_when_caller_default_device_is_meta(self):
        with torch.device("meta"):
            self.assertTrue(audit_transformer_loss_gradient().passed)

    def test_caller_cpu_rng_is_preserved_on_success(self):
        state = torch.get_rng_state().clone()
        audit_transformer_loss_gradient()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_caller_cpu_rng_is_preserved_on_failure(self):
        state = torch.get_rng_state().clone()
        with patch(
            "ai_journey.transformer_gradient_audit.manual_cross_entropy",
            side_effect=RuntimeError("probe"),
        ):
            with self.assertRaises(RuntimeError):
                audit_transformer_loss_gradient()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_incorrect_loss_derivative_fails_parameter_gate(self):
        def broken(logits, labels):
            result = manual_cross_entropy(logits, labels)
            result.dlogits[:] = 0
            return result

        with patch(
            "ai_journey.transformer_gradient_audit.manual_cross_entropy",
            side_effect=broken,
        ):
            report = audit_transformer_loss_gradient()
        self.assertFalse(report.passed)
        self.assertGreater(max(report.parameter_errors.values()), 1e-3)

    def test_rejects_invalid_controls(self):
        for kwargs in [
            {"seed": True},
            {"seed": -1},
            {"seed": 2**32},
            {"seed": 0.5},
            {"tolerance": -1},
            {"tolerance": np.nan},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                audit_transformer_loss_gradient(**kwargs)


if __name__ == "__main__":
    unittest.main()
