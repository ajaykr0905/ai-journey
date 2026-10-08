from __future__ import annotations

import unittest
from hashlib import sha256

from ai_journey.gpt_bigram import (
    CorpusSource,
    GPTBigramError,
    TINY_SHAKESPEARE,
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


if __name__ == "__main__":
    unittest.main()
