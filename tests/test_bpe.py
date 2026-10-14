import unittest

from ai_journey.bpe import pair_counts, merge_pair, TokenizerError, ByteTokenizer


class PairPrimitiveTests(unittest.TestCase):
    def test_overlap_counts_and_nonoverlapping_merge(self):
        stream = [97] * 5
        self.assertEqual(pair_counts(stream)[(97, 97)], 4)
        self.assertEqual(merge_pair(stream, (97, 97), 256), (256, 256, 97))
        self.assertEqual(stream, [97] * 5)

    def test_empty_singleton_and_invalid_ids(self):
        self.assertFalse(pair_counts([]))
        self.assertEqual(merge_pair([1], (1, 2), 256), (1,))
        for values in ([True], [-1], [1.0]):
            with self.assertRaises(TokenizerError):
                pair_counts(values)


class ByteTokenizerTests(unittest.TestCase):
    def test_deterministic_training_and_multilingual_roundtrip(self):
        corpus = "hello नमस्ते தமிழ் 😀\n" * 30
        first = ByteTokenizer.train(corpus, vocab_size=300)
        second = ByteTokenizer.train(corpus, vocab_size=300)
        self.assertEqual(first.merges, second.merges)
        self.assertLess(len(first.encode(corpus)), len(corpus.encode()))
        for text in ("", corpus, "नया text Ω\x00"):
            self.assertEqual(first.decode(first.encode(text)), text)

    def test_invalid_merge_graphs_and_budget_controls(self):
        for merges in (((256, 1),), ((1, 2), (1, 2)), ((True, 1),)):
            with self.assertRaises(TokenizerError):
                ByteTokenizer(merges)
        with self.assertRaises(TokenizerError):
            ByteTokenizer.train("text", vocab_size=True)
        self.assertEqual(ByteTokenizer.train("x").vocab_size, 256)
        with self.assertRaises(TokenizerError):
            ByteTokenizer().decode_bytes([256])
