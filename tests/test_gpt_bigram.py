from __future__ import annotations

import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_journey.gpt_bigram import (
    CharacterVocabulary,
    CorpusSource,
    GPTBigramError,
    TINY_SHAKESPEARE,
    load_verified_corpus,
    tokenize_and_split,
    validate_corpus_bytes,
)


def source_for(payload: bytes) -> CorpusSource:
    return CorpusSource(
        "fixture",
        "https://example.test/fixture.txt",
        sha256(payload).hexdigest(),
        len(payload),
        "CC0-1.0",
    )


class CorpusSourceTests(unittest.TestCase):
    def test_tiny_shakespeare_source_is_content_addressed(self) -> None:
        self.assertEqual(TINY_SHAKESPEARE.byte_count, 1_115_394)
        self.assertEqual(len(TINY_SHAKESPEARE.sha256), 64)
        self.assertTrue(TINY_SHAKESPEARE.url.startswith("https://"))

    def test_source_rejects_unpinned_or_insecure_metadata(self) -> None:
        with self.assertRaisesRegex(GPTBigramError, "HTTPS"):
            CorpusSource("corpus", "http://example.test/data", "0" * 64, 1, "CC0")
        with self.assertRaisesRegex(GPTBigramError, "sha256"):
            CorpusSource("corpus", "https://example.test/data", "ABC", 1, "CC0")
        with self.assertRaisesRegex(GPTBigramError, "positive"):
            CorpusSource("corpus", "https://example.test/data", "0" * 64, 0, "CC0")


class CorpusBytesTests(unittest.TestCase):
    def test_validates_and_decodes_exact_public_bytes(self) -> None:
        payload = "First Citizen:\nSpeak.\n".encode()
        self.assertEqual(
            validate_corpus_bytes(payload, source_for(payload)), payload.decode()
        )

    def test_rejects_truncation_and_content_substitution(self) -> None:
        payload = b"abcd"
        source = source_for(payload)
        with self.assertRaisesRegex(GPTBigramError, "byte count mismatch"):
            validate_corpus_bytes(payload[:-1], source)
        with self.assertRaisesRegex(GPTBigramError, "sha256 mismatch"):
            validate_corpus_bytes(b"abce", source)

    def test_loads_only_a_matching_local_snapshot(self) -> None:
        payload = b"To be, or not to be.\n"
        with TemporaryDirectory() as directory:
            path = Path(directory, "input.txt")
            path.write_bytes(payload)
            self.assertEqual(
                load_verified_corpus(path, source_for(payload)), payload.decode()
            )
            path.write_bytes(payload + b"changed")
            with self.assertRaisesRegex(GPTBigramError, "byte count mismatch"):
                load_verified_corpus(path, source_for(payload))


class CharacterVocabularyTests(unittest.TestCase):
    def test_round_trips_text_with_stable_sorted_ids(self) -> None:
        vocabulary = CharacterVocabulary.from_text("cab\nca")
        self.assertEqual(vocabulary.tokens, ("\n", "a", "b", "c"))
        encoded = vocabulary.encode("cab\n")
        self.assertEqual(encoded, (3, 1, 2, 0))
        self.assertEqual(vocabulary.decode(encoded), "cab\n")

    def test_rejects_unknown_characters_and_invalid_ids(self) -> None:
        vocabulary = CharacterVocabulary.from_text("abc")
        with self.assertRaisesRegex(GPTBigramError, "unknown character"):
            vocabulary.encode("abd")
        with self.assertRaisesRegex(GPTBigramError, "out of range"):
            vocabulary.decode([3])


class CorpusSplitTests(unittest.TestCase):
    def test_uses_a_contiguous_held_out_suffix(self) -> None:
        text = "abcdefghij"
        vocabulary = CharacterVocabulary.from_text(text)
        split = tokenize_and_split(text, vocabulary, validation_fraction=0.2)
        self.assertEqual(split.split_index, 8)
        self.assertEqual(vocabulary.decode(split.train.tolist()), "abcdefgh")
        self.assertEqual(vocabulary.decode(split.validation.tolist()), "ij")
        self.assertEqual(split.train.dtype, split.validation.dtype)

    def test_rejects_partitions_too_short_for_next_token_targets(self) -> None:
        vocabulary = CharacterVocabulary.from_text("abcd")
        with self.assertRaisesRegex(GPTBigramError, "at least two"):
            tokenize_and_split("abcd", vocabulary, validation_fraction=0.25)


if __name__ == "__main__":
    unittest.main()
