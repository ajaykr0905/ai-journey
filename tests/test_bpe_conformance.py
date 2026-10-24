import unittest

from ai_journey.bpe import ByteTokenizer, RegexTokenizer, TokenizerError
from ai_journey.bpe_conformance import build_training_audit, replay_merge_prefix
from ai_journey.tokenizer_io import tokenizer_fingerprint
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

    def test_distinguishes_overlapping_frequency_from_applied_merges(self):
        tokenizer = train_documents(("aaaaa",) * 3, vocab_size=257)
        (observation,) = replay_merge_prefix(("aaaaa",) * 3, tokenizer)
        self.assertEqual(observation.frequency, 12)
        self.assertEqual(observation.applied_count, 6)
        self.assertEqual(observation.tokens_before, 15)
        self.assertEqual(observation.tokens_after, 9)

    def test_records_exact_token_expansion_and_dependency_depth(self):
        tokenizer = ByteTokenizer(((97, 98), (256, 99), (257, 257)))
        observations = replay_merge_prefix(("abcabcabcabc",), tokenizer)
        self.assertEqual(
            [row.token_hex for row in observations],
            [b"ab".hex(), b"abc".hex(), b"abcabc".hex()],
        )
        self.assertEqual([row.byte_length for row in observations], [2, 3, 6])
        self.assertEqual([row.dependency_depth for row in observations], [1, 2, 3])

    def test_binds_audit_to_exact_corpus_and_tokenizer_identities(self):
        documents = ("banana banana", "नमस्ते नमस्ते")
        tokenizer = train_documents(documents, vocab_size=270)
        report = build_training_audit(documents, tokenizer)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["document_count"], 2)
        self.assertEqual(report["initial_tokens"], report["utf8_bytes"])
        self.assertLess(report["final_tokens"], report["initial_tokens"])
        self.assertEqual(
            report["compression_ratio"], report["final_tokens"] / report["utf8_bytes"]
        )
        self.assertEqual(report["tokenizer_sha256"], tokenizer_fingerprint(tokenizer))
        changed = build_training_audit(documents[::-1], tokenizer)
        self.assertNotEqual(report["corpus_sha256"], changed["corpus_sha256"])
        self.assertEqual(report["tokenizer_sha256"], changed["tokenizer_sha256"])


if __name__ == "__main__":
    unittest.main()
