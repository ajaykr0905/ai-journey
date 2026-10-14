import unittest
import torch

from ai_journey.bpe import TokenizerError
from ai_journey.bpe_corpus import build_bpe_corpus, EOS_MARKER
from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TransformerConfig,
)


class BPECorpusTests(unittest.TestCase):
    def test_byte_codec_trains_transformer_batches_and_preserves_eos(self):
        corpus = build_bpe_corpus(
            ("hello hello", "नमस्ते नमस्ते"), ("unseen 🙂",), vocab_size=270
        )
        self.assertEqual(int(corpus.train_tokens[-1]), corpus.vocab_size - 1)
        self.assertTrue(
            corpus.codec.decode(corpus.validation_tokens.tolist()).endswith(EOS_MARKER)
        )
        cursor = BatchCursor(corpus.train_tokens, block_size=4, batch_size=2)
        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=corpus.vocab_size,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
            )
        )
        x, y = cursor.next()
        _, loss = model(x, y)
        self.assertTrue(torch.isfinite(loss))

    def test_validation_cannot_affect_fitted_vocabulary_and_leakage_is_rejected(self):
        train = ("banana banana",)
        first = build_bpe_corpus(train, ("unseen",), vocab_size=270)
        second = build_bpe_corpus(train, ("未知",), vocab_size=270)
        self.assertEqual(first.codec.fingerprint(), second.codec.fingerprint())
        self.assertNotEqual(first.fingerprint(), second.fingerprint())
        with self.assertRaises(TokenizerError):
            build_bpe_corpus(train, train)
        with self.assertRaises(TokenizerError):
            build_bpe_corpus((EOS_MARKER,), ("held out",))
