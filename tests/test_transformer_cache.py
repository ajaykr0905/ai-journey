from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_journey.transformer_cache import DecoderCache, LayerKV
from ai_journey.transformer_lab import TransformerLabError


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


if __name__ == "__main__":
    unittest.main()
