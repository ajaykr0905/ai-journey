from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_journey.transformer_cache import DecoderCache, LayerKV, cached_attention
from ai_journey.transformer_lab import (
    CausalSelfAttention,
    TransformerConfig,
    TransformerLabError,
)


class CacheStateTests(unittest.TestCase):
    def test_states_own_storage_and_properties_do_not_alias(self) -> None:
        keys = torch.randn(2, 2, 3, 4, requires_grad=True)
        layer = LayerKV(keys, keys + 1)
        original = layer.keys
        with torch.no_grad():
            keys.zero_()
        layer.keys.zero_()
        torch.testing.assert_close(layer.keys, original)
        self.assertFalse(layer.keys.requires_grad)
        tokens = torch.ones(2, 3, dtype=torch.long)
        state = DecoderCache(
            tokens, (layer,), config_digest="a" * 64, model_digest="b" * 64
        )
        tokens.zero_()
        state.tokens.zero_()
        state.layers[0].keys.zero_()
        self.assertTrue(torch.all(state.tokens == 1))
        torch.testing.assert_close(state.layers[0].keys, original)

    def test_rejects_nonfinite_and_mismatched_layer_tensors(self) -> None:
        keys = torch.ones(2, 2, 3, 4)
        for values in (
            keys[:, :, :2],
            keys.double(),
            torch.full_like(keys, float("nan")),
        ):
            with self.assertRaises(TransformerLabError):
                LayerKV(keys, values)
        with self.assertRaises(TransformerLabError):
            LayerKV(keys.long(), keys.long())

    def test_rejects_invalid_decoder_identity_and_shape(self) -> None:
        layer = LayerKV(torch.ones(2, 2, 3, 4), torch.ones(2, 2, 3, 4))
        with self.assertRaises(TransformerLabError):
            DecoderCache(
                torch.zeros(2, 2, dtype=torch.long),
                (layer,),
                config_digest="a" * 64,
                model_digest="b" * 64,
            )
        with self.assertRaises(TransformerLabError):
            DecoderCache(
                torch.zeros(2, 3, dtype=torch.long),
                (layer,),
                config_digest="bad",
                model_digest="b" * 64,
            )


class CachedAttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.attention = CausalSelfAttention(
            TransformerConfig(
                vocab_size=11, block_size=8, embedding_dim=12, head_count=3, dropout=0.2
            )
        ).eval()

    def test_prefill_and_chunks_match_full_attention(self) -> None:
        inputs = torch.randn(2, 7, 12)
        expected = self.attention(inputs)
        for sizes in ((7,), (1, 1, 1, 1, 1, 1, 1), (3, 2, 2)):
            outputs = []
            cache = None
            offset = 0
            for size in sizes:
                output, cache = cached_attention(
                    self.attention, inputs[:, offset : offset + size], cache
                )
                outputs.append(output)
                offset += size
            torch.testing.assert_close(torch.cat(outputs, dim=1), expected)
            self.assertEqual(cache.keys.shape, (2, 3, 7, 4))

    def test_rejects_training_overflow_and_bad_inputs(self) -> None:
        inputs = torch.randn(2, 5, 12)
        self.attention.train()
        with self.assertRaisesRegex(TransformerLabError, "evaluation"):
            cached_attention(self.attention, inputs)
        self.attention.eval()
        _, cache = cached_attention(self.attention, inputs)
        with self.assertRaisesRegex(TransformerLabError, "block_size"):
            cached_attention(self.attention, inputs, cache)
        for invalid in (
            inputs.double(),
            inputs[:, :, :4],
            torch.full_like(inputs, float("nan")),
        ):
            with self.assertRaises(TransformerLabError):
                cached_attention(self.attention, invalid)
        with self.assertRaisesRegex(TransformerLabError, "mismatch"):
            cached_attention(self.attention, inputs[:1], cache)


if __name__ == "__main__":
    unittest.main()
