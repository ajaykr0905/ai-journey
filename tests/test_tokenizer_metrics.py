import unittest

from ai_journey.bpe import ByteTokenizer, TokenizerError
from ai_journey.tokenizer_metrics import compression_metrics


class CompressionTests(unittest.TestCase):
    def test_utf8_bytes_are_not_confused_with_unicode_characters(self):
        metrics = compression_metrics(ByteTokenizer(), ("a", "ह"))
        self.assertEqual(metrics.utf8_bytes, 4)
        self.assertEqual(metrics.characters, 2)
        self.assertEqual(metrics.tokens, 4)
        self.assertEqual(metrics.bytes_per_token, 1)
        self.assertEqual(metrics.characters_per_token, 0.5)
        self.assertEqual(metrics.byte_fallback_tokens, 4)

    def test_trained_tokens_and_document_boundaries_have_exact_denominators(self):
        tokenizer = ByteTokenizer(((97, 98),))
        metrics = compression_metrics(tokenizer, ("ab", "ab"))
        self.assertEqual(metrics.tokens, 2)
        self.assertEqual(metrics.used_vocabulary, 1)
        self.assertEqual(metrics.byte_fallback_tokens, 0)
        self.assertEqual(metrics.token_entropy_bits, 0)
        with self.assertRaises(TokenizerError):
            compression_metrics(tokenizer, ("",))
