import unittest

from ai_journey.parallel_corpus import (
    extract_sentences,
    Sentence,
    align_and_split,
    parallel_folds,
)
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


class ParallelSplitTests(unittest.TestCase):
    def test_translations_keep_matching_disjoint_ids_independent_of_file_order(self):
        first = tuple(Sentence(str(index), f"first {index}") for index in range(10))
        second = tuple(Sentence(str(index), f"second {index}") for index in range(10))
        split = align_and_split(first, second, train_count=7)
        self.assertEqual(
            split, align_and_split(first[::-1], second[::-1], train_count=7)
        )
        self.assertEqual(
            [row.sentence_id for row in split.first_train],
            [row.sentence_id for row in split.second_train],
        )
        self.assertFalse(
            {row.sentence_id for row in split.first_train}
            & {row.sentence_id for row in split.first_validation}
        )

    def test_unpaired_or_duplicate_ids_are_rejected(self):
        first = (Sentence("a", "one"), Sentence("b", "two"))
        for second in (first[:1], (first[0], first[0])):
            with self.assertRaises(TokenizerError):
                align_and_split(first, second, train_count=1)

    def test_balanced_cross_validation_visits_every_translation_exactly_once(self):
        first = tuple(Sentence(str(index), f"first {index}") for index in range(23))
        second = tuple(Sentence(str(index), f"second {index}") for index in range(23))
        folds = parallel_folds(first, second, fold_count=10)
        self.assertEqual(
            folds, parallel_folds(first[::-1], second[::-1], fold_count=10)
        )
        seen = []
        sizes = []
        for split in folds:
            heldout = [row.sentence_id for row in split.first_validation]
            seen.extend(heldout)
            sizes.append(len(heldout))
            self.assertEqual(
                heldout, [row.sentence_id for row in split.second_validation]
            )
            self.assertFalse(
                set(heldout) & {row.sentence_id for row in split.first_train}
            )
        self.assertEqual(sorted(seen), sorted(row.sentence_id for row in first))
        self.assertLessEqual(max(sizes) - min(sizes), 1)
