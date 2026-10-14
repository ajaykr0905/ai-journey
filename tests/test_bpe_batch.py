import unittest

import torch

from ai_journey.bpe import ByteTokenizer, TokenizerError
from ai_journey.token_batch import encode_batch, decode_batch
from ai_journey.transformer_lab import DecoderLanguageModel, TransformerConfig


class TokenBatchTests(unittest.TestCase):
    def test_padding_does_not_discard_real_zero_byte_tokens(self):
        tokenizer = ByteTokenizer()
        batch = encode_batch(tokenizer, ("\x00", "longer"))
        self.assertEqual(decode_batch(tokenizer, batch), ("\x00", "longer"))
        self.assertEqual(
            batch.valid[0].tolist(), [True, False, False, False, False, False]
        )
        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=256,
                block_size=8,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
            )
        )
        logits, _ = model(batch.token_ids, lengths=batch.lengths)
        self.assertTrue(torch.equal(logits[0, 1:], torch.zeros_like(logits[0, 1:])))

    def test_truncation_requires_an_explicit_direction(self):
        tokenizer = ByteTokenizer()
        with self.assertRaises(TokenizerError):
            encode_batch(tokenizer, ("abcd",), max_tokens=2)
        self.assertEqual(
            decode_batch(
                tokenizer,
                encode_batch(tokenizer, ("abcd",), max_tokens=2, truncation="left"),
            ),
            ("cd",),
        )
        self.assertEqual(
            decode_batch(
                tokenizer,
                encode_batch(tokenizer, ("abcd",), max_tokens=2, truncation="right"),
            ),
            ("ab",),
        )
