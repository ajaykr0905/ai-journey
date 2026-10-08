from __future__ import annotations

import unittest

from ai_journey.gpt_bigram import CorpusSource, GPTBigramError, TINY_SHAKESPEARE


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


if __name__ == "__main__":
    unittest.main()
