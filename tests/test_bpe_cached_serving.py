import unittest
import torch

from ai_journey.bpe_corpus import build_bpe_corpus
from ai_journey.bpe_serving import generate_utf8
from ai_journey.transformer_lab import (
    DecoderLanguageModel,
    TransformerConfig,
    model_fingerprint,
)


class CachedUtf8ServingTests(unittest.TestCase):
    def test_greedy_and_seeded_cache_rollover_match_full_forward_exactly(self):
        corpus = build_bpe_corpus(
            ("hello hello दुनिया नमस्ते",), ("held out",), vocab_size=270
        )
        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=corpus.vocab_size,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
            )
        ).double()
        fingerprint = model_fingerprint(model)
        rng = torch.get_rng_state().clone()
        for prompt in ("hello", "नमस्ते दुनिया", "🙂"):
            for top_k in (1, 20):
                full = generate_utf8(
                    model, corpus.codec, prompt, new_tokens=12, top_k=top_k
                )
                cached = generate_utf8(
                    model, corpus.codec, prompt, new_tokens=12, top_k=top_k, cached=True
                )
                self.assertEqual(full.pop("backend"), "full-forward")
                self.assertEqual(cached.pop("backend"), "kv-cache")
                self.assertEqual(full, cached)
        self.assertEqual(model_fingerprint(model), fingerprint)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)

    def test_invalid_backend_control_fails_without_mode_changes(self):
        corpus = build_bpe_corpus(("abab",), ("held out",))
        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=corpus.vocab_size,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
            )
        )
        with self.assertRaises(TypeError):
            generate_utf8(model, corpus.codec, "hello", cached=1)
        self.assertTrue(model.training)
