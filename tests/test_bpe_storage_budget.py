import unittest

from ai_journey.bpe import ByteTokenizer, TokenizerError


class VocabularyBudgetTests(unittest.TestCase):
    def test_small_merge_table_cannot_request_exponential_byte_expansion(self):
        pairs = []
        previous = 97
        for index in range(30):
            pairs.append((previous, previous))
            previous = 256 + index
        with self.assertRaisesRegex(TokenizerError, "byte-storage budget"):
            ByteTokenizer(tuple(pairs))

    def test_cumulative_storage_is_bounded_even_when_each_token_is_small_enough(self):
        pairs = []
        previous = 97
        for index in range(13):
            pairs.append((previous, previous))
            previous = 256 + index
        for _ in range(1500):
            pairs.append((previous, 98))
            previous = 256 + len(pairs) - 1
        with self.assertRaisesRegex(TokenizerError, "byte-storage budget"):
            ByteTokenizer(tuple(pairs))

    def test_normal_multilingual_training_remains_lossless(self):
        text = "hello नमस्ते 🙂 " * 20
        tokenizer = ByteTokenizer.train(text, vocab_size=300)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
