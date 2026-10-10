from __future__ import annotations

import sys
import copy
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_journey.transformer_cache import (
    DecoderCache,
    LayerKV,
    cached_attention,
    decode,
    prefill,
    migrate_cache,
)
from ai_journey.transformer_lab import (
    CausalSelfAttention,
    DecoderLanguageModel,
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


class CachedDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=11,
                block_size=8,
                embedding_dim=12,
                head_count=3,
                layer_count=2,
                dropout=0.25,
            )
        ).eval()

    def test_full_decoder_prefill_and_chunked_logits_agree(self) -> None:
        tokens = torch.randint(0, 11, (2, 7))
        expected, _ = self.model(tokens)
        first, cache = prefill(self.model, tokens[:, :2])
        middle, cache = decode(self.model, tokens[:, 2:5], cache)
        final, cache = decode(self.model, tokens[:, 5:], cache)
        torch.testing.assert_close(torch.cat((first, middle, final), dim=1), expected)
        torch.testing.assert_close(cache.tokens, tokens)
        self.assertEqual(len(cache.layers), 2)
        self.assertFalse(final.requires_grad)

    def test_rejects_stale_model_and_invalid_decode_before_state_change(self) -> None:
        tokens = torch.randint(0, 11, (2, 3))
        _, cache = prefill(self.model, tokens)
        with self.assertRaisesRegex(TransformerLabError, "batch"):
            decode(self.model, tokens[:1, :1], cache)
        with torch.no_grad():
            self.model.lm_head.weight[0, 0] += 0.1
        with self.assertRaisesRegex(TransformerLabError, "modified model"):
            decode(self.model, tokens[:, :1], cache)
        torch.testing.assert_close(cache.tokens, tokens)

    def test_decoder_requires_eval_and_valid_prompt(self) -> None:
        self.model.train()
        with self.assertRaisesRegex(TransformerLabError, "evaluation"):
            prefill(self.model, torch.ones(2, 3, dtype=torch.long))
        self.model.eval()
        for tokens in (
            torch.ones(2, 3),
            torch.ones(2, 0, dtype=torch.long),
            torch.full((2, 1), 11, dtype=torch.long),
        ):
            with self.assertRaises((TransformerLabError, TypeError)):
                prefill(self.model, tokens)

    def test_rollover_resets_positions_and_matches_cropped_model(self) -> None:
        tokens = torch.randint(0, 11, (2, 17))
        _, cache = prefill(self.model, tokens[:, :6])
        actual, final = decode(self.model, tokens[:, 6:], cache)
        expected = []
        for end in range(7, 18):
            logits, _ = self.model(tokens[:, max(0, end - 8) : end])
            expected.append(logits[:, -1:])
        torch.testing.assert_close(actual, torch.cat(expected, dim=1))
        torch.testing.assert_close(final.tokens, tokens[:, -8:])
        torch.testing.assert_close(cache.tokens, tokens[:, :6])

    def test_single_token_rollover_agrees_with_chunk_rollover(self) -> None:
        tokens = torch.randint(0, 11, (1, 13))
        _, initial = prefill(self.model, tokens[:, :8])
        expected, expected_state = decode(self.model, tokens[:, 8:], initial)
        outputs = []
        state = initial
        for token in tokens[:, 8:].split(1, dim=1):
            output, state = decode(self.model, token, state)
            outputs.append(output)
        torch.testing.assert_close(torch.cat(outputs, dim=1), expected)
        torch.testing.assert_close(state.tokens, expected_state.tokens)

    def test_explicit_dtype_migration_rebuilds_for_converted_weights(self) -> None:
        tokens = torch.randint(0, 11, (2, 4))
        _, cache = prefill(self.model, tokens)
        converted = copy.deepcopy(self.model).double()
        state = migrate_cache(cache, self.model, converted)
        self.assertEqual(state.layers[0].keys.dtype, torch.float64)
        next_tokens = torch.randint(0, 11, (2, 2))
        actual, _ = decode(converted, next_tokens, state)
        expected, _ = converted(torch.cat((tokens, next_tokens), dim=1))
        torch.testing.assert_close(actual, expected[:, -2:])
        self.assertEqual(cache.layers[0].keys.dtype, torch.float32)
        with torch.no_grad():
            converted.lm_head.weight[0, 0] += 0.1
        with self.assertRaisesRegex(TransformerLabError, "conversion"):
            migrate_cache(cache, self.model, converted)

    def test_bfloat16_cache_binding_does_not_require_numpy_dtype_support(self) -> None:
        model = copy.deepcopy(self.model).bfloat16()
        tokens = torch.randint(0, 11, (1, 4))
        _, cache = prefill(model, tokens[:, :3])
        actual, _ = decode(model, tokens[:, 3:], cache)
        expected, _ = model(tokens)
        torch.testing.assert_close(actual, expected[:, -1:], rtol=0.02, atol=0.002)


if __name__ == "__main__":
    unittest.main()
