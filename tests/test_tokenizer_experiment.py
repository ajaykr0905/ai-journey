from dataclasses import replace
import unittest

from ai_journey.bpe import TokenizerError
from ai_journey.parallel_corpus import Sentence, align_and_split
from ai_journey.tokenizer_experiment import (
    run_compression_experiment,
    run_compression_cross_validation,
)


class ControlledCompressionTests(unittest.TestCase):
    def split(self):
        hi = tuple(Sentence(str(index), f"नमस्ते नमस्ते {index}") for index in range(8))
        en = tuple(Sentence(str(index), f"hello hello {index}") for index in range(8))
        return align_and_split(hi, en, train_count=6)

    def test_three_controlled_sizes_and_cross_language_holdouts_are_reproducible(self):
        first = run_compression_experiment(self.split(), vocab_sizes=(258, 260, 262))
        second = run_compression_experiment(self.split(), vocab_sizes=(258, 260, 262))
        self.assertEqual(len(first["evaluations"]), 6)
        for rows in (first["evaluations"], second["evaluations"]):
            for row in rows:
                self.assertGreaterEqual(row.pop("training_seconds"), 0)
                self.assertEqual(len(row["heldout"]), 2)
        self.assertEqual(first, second)
        self.assertEqual(first["byte_baseline"][0]["bytes_per_token"], 1)

    def test_misaligned_and_overlapping_holdouts_fail_before_training(self):
        split = self.split()
        for bad in (
            replace(split, second_train=split.second_train[::-1]),
            replace(
                split,
                first_validation=split.first_train,
                second_validation=split.second_train,
            ),
        ):
            with self.assertRaises(TokenizerError):
                run_compression_experiment(bad)

    def test_ten_fold_aggregation_uses_token_weighted_ratios(self):
        hi = tuple(Sentence(str(index), "नमस्ते " * (index + 1)) for index in range(10))
        en = tuple(Sentence(str(index), "hello " * (index + 1)) for index in range(10))
        report = run_compression_cross_validation(hi, en, vocab_sizes=(258, 260, 262))
        self.assertEqual(len(report["folds"]), 10)
        for trial, aggregate in enumerate(report["aggregate"]):
            metrics = aggregate["heldout"][0]
            self.assertEqual(metrics["documents"], 10)
            self.assertEqual(
                metrics["utf8_bytes"], sum(len(row.text.encode()) for row in hi)
            )
            expected_tokens = sum(
                fold["evaluations"][trial]["heldout"][0]["tokens"]
                for fold in report["folds"]
            )
            self.assertEqual(metrics["tokens"], expected_tokens)
            self.assertEqual(
                metrics["bytes_per_token"], metrics["utf8_bytes"] / expected_tokens
            )
