import unittest

from ai_journey.bpe import ByteTokenizer, RegexTokenizer, TokenizerError
from ai_journey.bpe_conformance import (
    audit_fingerprint,
    build_training_audit,
    normalization_collisions,
    replay_merge_prefix,
    verify_training_audit,
)
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
        self.assertEqual(report["audit_sha256"], audit_fingerprint(report))
        tampered = {**report, "final_tokens": report["final_tokens"] + 1}
        self.assertNotEqual(tampered["audit_sha256"], audit_fingerprint(tampered))
        changed = build_training_audit(documents[::-1], tokenizer)
        self.assertNotEqual(report["corpus_sha256"], changed["corpus_sha256"])
        self.assertEqual(report["tokenizer_sha256"], changed["tokenizer_sha256"])

    def test_verifier_replays_sources_instead_of_trusting_a_recomputed_seal(self):
        documents = ("ababab", "abab")
        tokenizer = train_documents(documents, vocab_size=257)
        report = build_training_audit(documents, tokenizer)
        result = verify_training_audit(report, documents, tokenizer)
        self.assertEqual(result["verified_merges"], 1)

        damaged = {**report, "final_tokens": report["final_tokens"] + 1}
        with self.assertRaisesRegex(TokenizerError, "fingerprint"):
            verify_training_audit(damaged, documents, tokenizer)
        damaged["audit_sha256"] = audit_fingerprint(damaged)
        with self.assertRaisesRegex(TokenizerError, "source replay"):
            verify_training_audit(damaged, documents, tokenizer)
        with self.assertRaisesRegex(TokenizerError, "source replay"):
            verify_training_audit(report, documents[::-1], tokenizer)

    def test_explains_why_training_stopped(self):
        limited = build_training_audit(
            ("ababab", "abab"),
            train_documents(("ababab", "abab"), vocab_size=257),
        )
        self.assertEqual(limited["stop_reason"], "vocabulary_limit")
        self.assertGreaterEqual(limited["next_frequency"], 2)
        self.assertIsNotNone(limited["next_pair"])

        infrequent = build_training_audit(("ab",), ByteTokenizer())
        self.assertEqual(infrequent["stop_reason"], "min_frequency")
        self.assertEqual(infrequent["next_frequency"], 1)

        converged = build_training_audit(("a", "b"), ByteTokenizer())
        self.assertEqual(converged["stop_reason"], "no_pairs")
        self.assertEqual(converged["next_frequency"], 0)
        self.assertIsNone(converged["next_pair"])

    def test_reports_dead_intermediate_tokens_and_byte_fallback_usage(self):
        documents = ("abababababababab",)
        tokenizer = train_documents(documents, vocab_size=259)
        report = build_training_audit(documents, tokenizer)
        self.assertEqual(tokenizer.merges, ((97, 98), (256, 256), (257, 257)))
        self.assertEqual(report["unused_learned_token_ids"], [256, 257])
        self.assertEqual(report["used_vocabulary"], 1)
        self.assertEqual(report["byte_fallback_tokens"], 0)
        self.assertEqual(report["token_frequencies"], [{"token_id": 258, "count": 2}])

        fallback = build_training_audit(("xyz",), ByteTokenizer())
        self.assertEqual(fallback["used_vocabulary"], 3)
        self.assertEqual(fallback["byte_fallback_tokens"], 3)

    def test_reconciles_encoded_token_lengths_to_every_source_byte(self):
        documents = ("hello hello", "नमस्ते🙂")
        tokenizer = train_documents(documents, vocab_size=270)
        report = build_training_audit(documents, tokenizer)
        accounted = sum(
            row["bytes"] * row["count"] for row in report["token_byte_length_histogram"]
        )
        self.assertEqual(accounted, report["utf8_bytes"])
        self.assertEqual(
            report["max_token_bytes"],
            max(row["bytes"] for row in report["token_byte_length_histogram"]),
        )

    def test_surfaces_normalization_collisions_without_changing_source_bytes(self):
        documents = (
            "caf\N{LATIN SMALL LETTER E WITH ACUTE}",
            "cafe\N{COMBINING ACUTE ACCENT}",
            "\N{LATIN SMALL LIGATURE FI}",
            "fi",
        )
        self.assertEqual(normalization_collisions(documents), ((0, 1),))
        self.assertEqual(
            normalization_collisions(documents, form="NFKC"), ((0, 1), (2, 3))
        )
        report = build_training_audit(documents, ByteTokenizer())
        self.assertEqual(report["nfc_collision_groups"], ((0, 1),))
        self.assertEqual(report["nfkc_collision_groups"], ((0, 1), (2, 3)))
        with self.assertRaisesRegex(TokenizerError, "normalization form"):
            normalization_collisions(documents, form="NFD")


if __name__ == "__main__":
    unittest.main()
