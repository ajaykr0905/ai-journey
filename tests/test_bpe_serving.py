import unittest
from unittest.mock import patch

import torch

from ai_journey.bpe import ByteTokenizer, SpecialTokenizer
from ai_journey.bpe_corpus import EOS_MARKER
from ai_journey.bpe_serving import generate_utf8, utf8_candidates
from ai_journey.tokenizer_codec import BPECodec
from ai_journey.transformer_lab import DecoderLanguageModel, TransformerConfig


class Utf8ServingTests(unittest.TestCase):
    def setUp(self):
        self.codec = BPECodec(SpecialTokenizer(ByteTokenizer(), (EOS_MARKER,)))
        self.model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=257,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
            )
        )

    def test_invalid_bytes_and_unfinishable_prefixes_are_masked(self):
        candidates = utf8_candidates(self.codec.tokenizer, b"", 1)
        self.assertNotIn(255, candidates)
        self.assertNotIn(0xE0, candidates)
        self.assertIn(0xC2, candidates)
        candidates = utf8_candidates(self.codec.tokenizer, b"\xe0\xa0", 0)
        self.assertIn(0x80, candidates)
        self.assertNotIn(97, candidates)

    def test_generation_rejects_invalid_bytes_and_restores_modes_and_rng(self):
        before = torch.get_rng_state().clone()

        def logits(context):
            values = torch.zeros(1, context.shape[1], 257)
            values[..., 255] = 100
            values[..., 0xE0] = 90
            values[..., 97] = 80
            return values, None

        with patch.object(self.model, "forward", side_effect=logits):
            result = generate_utf8(
                self.model, self.codec, "नमस्ते", new_tokens=2, top_k=1
            )
        self.assertEqual(result["text"], "नमस्तेaa")
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_eos_stops_without_rendering_a_fake_marker(self):
        def logits(context):
            values = torch.zeros(1, context.shape[1], 257)
            values[..., 256] = 100
            return values, None

        with patch.object(self.model, "forward", side_effect=logits):
            result = generate_utf8(self.model, self.codec, "hello", top_k=1)
        self.assertEqual(result["text"], "hello")
        self.assertTrue(result["eos"])
        self.assertEqual(result["token_ids"][-1], 256)
