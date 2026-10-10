from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_journey.transformer_cache import cache_from_bytes, cache_to_bytes, prefill
from ai_journey.transformer_lab import (
    DecoderLanguageModel,
    TransformerConfig,
    TransformerLabError,
)


class CacheSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=11,
                block_size=8,
                embedding_dim=12,
                head_count=3,
                layer_count=2,
            )
        ).eval()
        self.tokens = torch.randint(0, 11, (2, 4))
        _, self.cache = prefill(self.model, self.tokens)

    def test_snapshot_is_deterministic_portable_and_storage_owned(self) -> None:
        encoded = cache_to_bytes(self.cache)
        self.assertEqual(encoded, cache_to_bytes(self.cache))
        restored = cache_from_bytes(encoded)
        torch.testing.assert_close(restored.tokens, self.tokens)
        for actual, expected in zip(restored.layers, self.cache.layers, strict=True):
            torch.testing.assert_close(actual.keys, expected.keys)
            torch.testing.assert_close(actual.values, expected.values)
        self.assertEqual(restored.model_digest, self.cache.model_digest)

    def test_bfloat16_snapshot_roundtrips_raw_bytes(self) -> None:
        self.model.bfloat16()
        _, cache = prefill(self.model, self.tokens)
        restored = cache_from_bytes(cache_to_bytes(cache))
        torch.testing.assert_close(restored.layers[0].keys, cache.layers[0].keys)

    def test_corruption_unknown_versions_and_untrusted_formats_are_rejected(
        self,
    ) -> None:
        envelope = json.loads(cache_to_bytes(self.cache))
        envelope["payload"]["model_digest"] = "a" * 64
        with self.assertRaisesRegex(TransformerLabError, "checksum"):
            cache_from_bytes(json.dumps(envelope).encode())
        envelope = json.loads(cache_to_bytes(self.cache))
        envelope["version"] = True
        with self.assertRaisesRegex(TransformerLabError, "version"):
            cache_from_bytes(json.dumps(envelope).encode())
        for invalid in (
            b"\x80\x04pickle",
            b'{"version":1,"version":1}',
            b"[1]",
            b"{",
            b"\xff",
        ):
            with self.assertRaises(TransformerLabError):
                cache_from_bytes(invalid)


if __name__ == "__main__":
    unittest.main()
