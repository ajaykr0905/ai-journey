import unittest

from ai_journey.bpe import ByteTokenizer, TokenizerError
from ai_journey.tokenizer_training import train_documents


class DocumentTrainingTests(unittest.TestCase):
    def test_training_never_invents_pairs_across_document_boundaries(self):
        tokenizer = train_documents(("a", "b", "a", "b"), vocab_size=257)
        self.assertEqual(tokenizer.merges, ())
        self.assertEqual(
            ByteTokenizer.train("abab", vocab_size=257).merges, ((97, 98),)
        )

    def test_regex_training_is_deterministic_under_document_permutation(self):
        documents = ("hello world", "hello world", "नमस्ते दुनिया")
        first = train_documents(documents, pattern=r"\w+|[^\w\s]+|\s+", vocab_size=270)
        second = train_documents(documents[::-1], pattern=first.pattern, vocab_size=270)
        self.assertEqual(first.merges, second.merges)
        for text in documents:
            self.assertEqual(first.decode(first.encode(text)), text)
        with self.assertRaises(TokenizerError):
            train_documents(("",))
