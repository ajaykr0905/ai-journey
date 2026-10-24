import unittest

from ai_journey.bpe import ByteTokenizer, RegexTokenizer, TokenizerError
from ai_journey.bpe_conformance import replay_merge_prefix
from ai_journey.tokenizer_training import train_documents


class MergeReplayTests(unittest.TestCase):
    def test_replays_byte_and_boundary_aware_training_exactly(self):
        documents = ("banana banana", "banana bandana", "नमस्ते नमस्ते")
        for pattern in (None, r"\w+|[^\w\s]+|\s+"):
            tokenizer = train_documents(
                documents, vocab_size=276, min_frequency=2, pattern=pattern
            )
            observations = replay_merge_prefix(documents, tokenizer)
            self.assertEqual(tuple(row.pair for row in observations), tokenizer.merges)
            self.assertTrue(all(row.frequency >= 2 for row in observations))

    def test_rejects_syntactically_valid_noncanonical_merge_tables(self):
        documents = ("ababab", "abab")
        canonical = train_documents(documents, vocab_size=257)
        self.assertEqual(canonical.merges, ((97, 98),))
        with self.assertRaisesRegex(TokenizerError, "canonical"):
            replay_merge_prefix(documents, ByteTokenizer(((98, 97),)))
        with self.assertRaisesRegex(TokenizerError, "frequency"):
            replay_merge_prefix(
                documents,
                RegexTokenizer(((97, 98),), pattern=r"\w+|[^\w\s]+|\s+"),
                min_frequency=100,
            )


if __name__ == "__main__":
    unittest.main()
