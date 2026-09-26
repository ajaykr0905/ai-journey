from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.context_mlp import ContextMLPError
from ai_journey.model_selection import (
    build_partitioned_datasets,
    minibatch_epochs,
    partition_fingerprints,
    split_train_dev_test,
    TrainingConfig,
)


class CorpusPartitionTests(unittest.TestCase):
    def test_three_way_split_is_deterministic_and_disjoint(self) -> None:
        words = tuple(f"name{letter}" for letter in "abcdefghij")
        first = split_train_dev_test(
            words, development_fraction=0.2, test_fraction=0.2, seed=17
        )
        second = split_train_dev_test(
            words, development_fraction=0.2, test_fraction=0.2, seed=17
        )

        self.assertEqual(first, second)
        self.assertEqual((len(first.train), len(first.development), len(first.test)), (6, 2, 2))
        self.assertFalse(set(first.train) & set(first.development))
        self.assertFalse(set(first.train) & set(first.test))
        self.assertFalse(set(first.development) & set(first.test))
        self.assertEqual(set(first.train + first.development + first.test), set(words))

    def test_three_way_split_rejects_invalid_allocations(self) -> None:
        with self.assertRaises(ContextMLPError):
            split_train_dev_test(("anna", "aria"))
        with self.assertRaises(ContextMLPError):
            split_train_dev_test(
                ("anna", "aria", "navi", "priya"),
                development_fraction=0.5,
                test_fraction=0.5,
            )

    def test_encoded_partitions_share_vocabulary_and_context_width(self) -> None:
        datasets = build_partitioned_datasets(
            ("anna", "aria", "navi", "devin", "priya", "samira"),
            block_size=4,
            development_fraction=0.2,
            test_fraction=0.2,
            seed=9,
        )

        self.assertEqual(datasets.train.block_size, 4)
        self.assertEqual(datasets.development.block_size, 4)
        self.assertEqual(datasets.test.block_size, 4)
        self.assertIs(datasets.train.vocabulary, datasets.development.vocabulary)
        self.assertIs(datasets.train.vocabulary, datasets.test.vocabulary)

    def test_partition_fingerprints_capture_each_encoded_split(self) -> None:
        first = build_partitioned_datasets(
            ("anna", "aria", "navi", "devin", "priya", "samira"), seed=9
        )
        second = build_partitioned_datasets(
            ("anna", "aria", "navi", "devin", "priya", "samira"), seed=9
        )

        fingerprints = partition_fingerprints(first)
        self.assertEqual(fingerprints, partition_fingerprints(second))
        self.assertEqual(len(fingerprints.train), 64)
        self.assertEqual(len({fingerprints.train, fingerprints.development, fingerprints.test}), 3)

    def test_epoch_batches_cover_every_sample_once_per_epoch(self) -> None:
        datasets = build_partitioned_datasets(
            tuple(f"name{letter}" for letter in "abcdefghijkl"), seed=3
        )
        batches = minibatch_epochs(datasets.train, batch_size=7, epochs=3, seed=21)

        self.assertEqual({batch.epoch for batch in batches}, {0, 1, 2})
        for epoch in range(3):
            epoch_batches = [batch for batch in batches if batch.epoch == epoch]
            self.assertEqual(
                sum(batch.dataset.sample_count for batch in epoch_batches),
                datasets.train.sample_count,
            )
            self.assertEqual(
                [batch.index for batch in epoch_batches],
                list(range(len(epoch_batches))),
            )


class TrainingConfigTests(unittest.TestCase):
    def test_config_rejects_invalid_hyperparameters(self) -> None:
        for keyword, value in (
            ("epochs", 0),
            ("batch_size", True),
            ("learning_rate", float("nan")),
            ("embedding_dim", -1),
            ("hidden_dim", 0),
            ("seed", False),
        ):
            with self.subTest(keyword=keyword), self.assertRaises((TypeError, ContextMLPError)):
                TrainingConfig(**{keyword: value})


if __name__ == "__main__":
    unittest.main()
