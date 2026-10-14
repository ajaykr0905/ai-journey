import unittest

from ai_journey.parallel_corpus import extract_sentences
from ai_journey.bpe import TokenizerError


class SentenceExtractionTests(unittest.TestCase):
    def test_metadata_text_is_preserved_instead_of_reconstructed_annotations(self):
        payload = "# sent_id = a\r\n# text = नमस्ते दुनिया।\r\n1\tignored\r\n\r\n# text = hello  world!\n# sent_id = b".encode()
        records = extract_sentences(payload)
        self.assertEqual(
            [(row.sentence_id, row.text) for row in records],
            [("a", "नमस्ते दुनिया।"), ("b", "hello  world!")],
        )

    def test_missing_or_ambiguous_metadata_is_rejected(self):
        for payload in (
            b"# text = text",
            b"# sent_id = a\n# text = \n",
            b"# sent_id = a\n# text = x\n# text = y",
            b"# sent_id = a\n# text = x\n\n# sent_id = a\n# text = y",
        ):
            with self.assertRaises(TokenizerError):
                extract_sentences(payload)
