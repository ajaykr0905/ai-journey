from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.context_mlp import ContextMLPError
from ai_journey.context_mlp import (
    evaluate_context_mlp,
    initialize_context_mlp,
    loss_and_gradients,
)
from ai_journey.model_selection import (
    apply_sgd,
    build_partitioned_datasets,
    minibatch_epochs,
    partition_fingerprints,
    split_train_dev_test,
    TrainingConfig,
    train_and_validate_context_mlp,
    train_minibatch_context_mlp,
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

    def test_sgd_update_reduces_loss_on_a_small_batch(self) -> None:
        datasets = build_partitioned_datasets(
            ("anna", "aria", "navi", "devin", "priya", "samira"), seed=3
        )
        model = initialize_context_mlp(datasets.train, seed=7)
        before, gradients = loss_and_gradients(datasets.train, model)
        updated = apply_sgd(model, gradients, learning_rate=0.1)
        after, _ = loss_and_gradients(datasets.train, updated)

        self.assertLess(after, before)

    def test_minibatch_training_is_reproducible_and_reduces_loss(self) -> None:
        datasets = build_partitioned_datasets(
            tuple(f"name{letter}" for letter in "abcdefghijkl"), seed=3
        )
        config = TrainingConfig(
            epochs=8,
            batch_size=16,
            learning_rate=0.1,
            embedding_dim=4,
            hidden_dim=16,
            seed=7,
        )
        first = train_minibatch_context_mlp(datasets.train, config)
        second = train_minibatch_context_mlp(datasets.train, config)

        self.assertEqual(first.training_nll, second.training_nll)
        self.assertLess(first.training_nll[-1], first.training_nll[0])

    def test_validation_loss_is_measured_after_every_epoch(self) -> None:
        datasets = build_partitioned_datasets(
            tuple(f"name{letter}" for letter in "abcdefghijkl"), seed=3
        )
        config = TrainingConfig(epochs=4, batch_size=16, hidden_dim=16, seed=7)
        result = train_and_validate_context_mlp(datasets, config)

        self.assertEqual(len(result.training_nll), 4)
        self.assertEqual(len(result.development_nll), 4)
        self.assertTrue(all(loss > 0 for loss in result.development_nll))

    def test_best_model_is_selected_by_development_loss(self) -> None:
        datasets = build_partitioned_datasets(
            tuple(f"name{letter}" for letter in "abcdefghijkl"), seed=3
        )
        result = train_and_validate_context_mlp(
            datasets,
            TrainingConfig(epochs=6, batch_size=16, hidden_dim=16, seed=7),
        )

        self.assertEqual(
            result.development_nll[result.best_epoch], min(result.development_nll)
        )
        self.assertAlmostEqual(
            evaluate_context_mlp(datasets.development, result.best_model).nll,
            min(result.development_nll),
        )


if __name__ == "__main__":
    unittest.main()
