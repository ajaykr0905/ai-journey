from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.transformer_lab import TransformerConfig, TransformerLabError


class TransformerConfigTests(unittest.TestCase):
    def test_config_validates_dimensions_and_has_stable_fingerprint(self) -> None:
        config = TransformerConfig(vocab_size=27, embedding_dim=24, head_count=3)
        self.assertEqual(config.head_dim, 8)
        self.assertEqual(config.fingerprint(), config.fingerprint())
        self.assertEqual(len(config.fingerprint()), 64)

    def test_config_rejects_incompatible_heads(self) -> None:
        with self.assertRaisesRegex(TransformerLabError, "divisible"):
            TransformerConfig(vocab_size=27, embedding_dim=10, head_count=3)


if __name__ == "__main__":
    unittest.main()
