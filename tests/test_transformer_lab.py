from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.transformer_lab import (
    CharacterCodec,
    TransformerConfig,
    TransformerLabError,
)


class TransformerConfigTests(unittest.TestCase):
    def test_config_validates_dimensions_and_has_stable_fingerprint(self) -> None:
        config = TransformerConfig(vocab_size=27, embedding_dim=24, head_count=3)
        self.assertEqual(config.head_dim, 8)
        self.assertEqual(config.fingerprint(), config.fingerprint())
        self.assertEqual(len(config.fingerprint()), 64)

    def test_config_rejects_incompatible_heads(self) -> None:
        with self.assertRaisesRegex(TransformerLabError, "divisible"):
            TransformerConfig(vocab_size=27, embedding_dim=10, head_count=3)


class CharacterCodecTests(unittest.TestCase):
    def test_codec_round_trips_public_text(self) -> None:
        codec = CharacterCodec.from_text("anna\naria\n")
        encoded = codec.encode("aria\n")
        self.assertEqual(codec.decode(list(encoded)), "aria\n")
        self.assertEqual(codec.tokens, tuple(sorted(set("anna\naria\n"))))

    def test_codec_rejects_unknown_characters(self) -> None:
        codec = CharacterCodec.from_text("ab")
        with self.assertRaisesRegex(TransformerLabError, "unknown"):
            codec.encode("abc")


if __name__ == "__main__":
    unittest.main()
