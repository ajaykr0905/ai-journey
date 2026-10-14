import unittest

from ai_journey.bpe import pair_counts, merge_pair, TokenizerError


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
