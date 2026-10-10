from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from ai_journey.transformer_certification import (
    TransformerCertificationConfig,
    certify_transformer,
)


class TransformerCertificationTests(unittest.TestCase):
    def test_certification_is_reproducible_and_rng_isolated(self):
        state = torch.get_rng_state().clone()
        first = certify_transformer()
        self.assertEqual(first, certify_transformer())
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertLess(first["cache_rollover_error"], 1e-10)
        self.assertLess(first["padding_error"], 1e-12)
        self.assertEqual(len(first["attention"]), 2)

    def test_invalid_controls_are_rejected_before_work(self):
        for values in (
            {"seed": True},
            {"seed": -1},
            {"tolerance": float("nan")},
            {"tolerance": True},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TransformerCertificationConfig(**values)
        default = TransformerCertificationConfig()
        with self.assertRaisesRegex(ValueError, "bounded"):
            replace(default, model=replace(default.model, block_size=65))

    def test_failed_measurement_prevents_certification(self):
        from ai_journey.attention_audit import AttentionCausalityAudit

        with patch(
            "ai_journey.transformer_certification.audit_attention_causality",
            return_value=AttentionCausalityAudit(1, 1, 4),
        ):
            with self.assertRaisesRegex(ValueError, "failed"):
                certify_transformer()
