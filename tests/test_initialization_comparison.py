from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.initialization_comparison import audit_model_initialization
from ai_journey.transformer_lab import (
    DecoderLanguageModel,
    TransformerConfig,
    seed_everything,
)


class InitializationAuditTests(unittest.TestCase):
    def test_audit_covers_every_initialized_matrix_in_module_order(self) -> None:
        seed_everything(26)
        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=17,
                block_size=4,
                embedding_dim=16,
                head_count=4,
                layer_count=1,
            )
        )
        audit = audit_model_initialization(model)
        expected_names = tuple(
            f"{name}.weight"
            for name, module in model.named_modules()
            if module.__class__.__name__ in {"Linear", "Embedding"}
        )
        self.assertEqual(tuple(item.name for item in audit), expected_names)
        self.assertTrue(all(item.observed_std > 0 for item in audit))
        self.assertTrue(
            all(item.bias_is_zero is not False for item in audit),
            audit,
        )

    def test_kaiming_audit_records_fan_in_gain_and_fixed_embeddings(self) -> None:
        gain = 1.25
        config = TransformerConfig(
            vocab_size=17,
            block_size=4,
            embedding_dim=16,
            head_count=4,
            layer_count=1,
            initialization_mode="kaiming_normal",
            initialization_gain=gain,
            initialization_std=0.03,
        )
        seed_everything(26)
        audit = audit_model_initialization(DecoderLanguageModel(config))
        by_name = {item.name: item for item in audit}
        linear = by_name["blocks.0.feed_forward.network.0.weight"]
        embedding = by_name["token_embedding.weight"]
        self.assertEqual(linear.fan_in, config.embedding_dim)
        self.assertEqual(
            linear.expected_std,
            gain / math.sqrt(config.embedding_dim),
        )
        self.assertIsNone(embedding.fan_in)
        self.assertEqual(embedding.expected_std, config.initialization_std)
        self.assertIsNone(embedding.bias_is_zero)

    def test_audit_rejects_unrelated_module(self) -> None:
        from torch import nn

        with self.assertRaisesRegex(TypeError, "DecoderLanguageModel"):
            audit_model_initialization(nn.Linear(2, 2))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
