from __future__ import annotations

import sys
import unittest
import json
from itertools import pairwise
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.context_mlp import ContextMLPError
from ai_journey.context_mlp import (
    evaluate_context_mlp,
    initialize_context_mlp,
    loss_and_gradients,
    model_fingerprint,
)
from ai_journey.model_selection import (
    apply_sgd,
    build_partitioned_datasets,
    detect_overfitting,
    evaluate_partitions,
    experiment_payload,
    learning_rate_grid,
    minibatch_epochs,
    partition_fingerprints,
    run_learning_rate_sweep,
    run_model_selection,
    select_best_trial,
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
            ("patience", 0),
            ("minimum_delta", -0.1),
            ("weight_decay", -0.1),
            ("max_gradient_norm", 0),
        ):
            with self.subTest(keyword=keyword), self.assertRaises((TypeError, ContextMLPError)):
                TrainingConfig(**{keyword: value})

    def test_learning_rate_grid_is_logarithmic_and_inclusive(self) -> None:
        rates = learning_rate_grid(0.001, 0.1, count=5)

        self.assertEqual(rates[0], 0.001)
        self.assertEqual(rates[-1], 0.1)
        ratios = [right / left for left, right in pairwise(rates)]
        self.assertTrue(all(abs(ratio - ratios[0]) < 1e-12 for ratio in ratios))
        with self.assertRaises(ContextMLPError):
            learning_rate_grid(0.1, 0.01, count=5)

    def test_learning_rate_sweep_runs_controlled_trials(self) -> None:
        datasets = build_partitioned_datasets(
            tuple(f"name{letter}" for letter in "abcdefghijkl"), seed=3
        )
        config = TrainingConfig(epochs=4, batch_size=16, hidden_dim=16, seed=7)
        first = run_learning_rate_sweep(datasets, config, (0.01, 0.05, 0.1))
        second = run_learning_rate_sweep(datasets, config, (0.01, 0.05, 0.1))

        self.assertEqual([trial.learning_rate for trial in first], [0.01, 0.05, 0.1])
        self.assertEqual(
            [trial.best_development_nll for trial in first],
            [trial.best_development_nll for trial in second],
        )

    def test_sweep_selection_uses_development_loss(self) -> None:
        datasets = build_partitioned_datasets(
            tuple(f"name{letter}" for letter in "abcdefghijkl"), seed=3
        )
        trials = run_learning_rate_sweep(
            datasets,
            TrainingConfig(epochs=4, batch_size=16, hidden_dim=16, seed=7),
            (0.01, 0.05, 0.1),
        )
        selected = select_best_trial(reversed(trials))

        self.assertEqual(
            selected.best_development_nll,
            min(trial.best_development_nll for trial in trials),
        )
        with self.assertRaises(ContextMLPError):
            select_best_trial(())

    def test_overfitting_requires_train_dev_divergence(self) -> None:
        signal = detect_overfitting(
            (2.0, 1.7, 1.4, 1.2, 1.0),
            (2.1, 1.8, 1.7, 1.9, 2.2),
            window=2,
        )

        self.assertTrue(signal.detected)
        self.assertLess(signal.training_change, 0)
        self.assertGreater(signal.development_change, 0)
        self.assertGreater(signal.final_gap, 0)
        self.assertFalse(
            detect_overfitting(
                (2.0, 1.8, 1.6, 1.4), (2.1, 1.9, 1.7, 1.5), window=2
            ).detected
        )

    def test_model_selection_experiment_is_reproducible(self) -> None:
        words = tuple(f"name{letter}" for letter in "abcdefghijkl")
        config = TrainingConfig(epochs=4, batch_size=16, hidden_dim=16, seed=7)
        first = run_model_selection(
            words, config=config, learning_rates=(0.01, 0.05, 0.1)
        )
        second = run_model_selection(
            words, config=config, learning_rates=(0.01, 0.05, 0.1)
        )

        self.assertEqual(first.fingerprints, second.fingerprints)
        self.assertEqual(
            first.selected.best_development_nll,
            second.selected.best_development_nll,
        )
        self.assertEqual(first.metrics, second.metrics)

    def test_experiment_payload_is_json_compatible_and_auditable(self) -> None:
        experiment = run_model_selection(
            tuple(f"name{letter}" for letter in "abcdefghijkl"),
            config=TrainingConfig(epochs=3, batch_size=16, hidden_dim=16, seed=7),
            learning_rates=(0.01, 0.05),
        )
        payload = experiment_payload(experiment)

        json.dumps(payload)
        self.assertEqual(len(payload["partition_fingerprints"]["train"]), 64)
        self.assertEqual(len(payload["selected_model_fingerprint"]), 64)
        self.assertEqual(len(payload["trials"]), 2)
        self.assertIn("test_gap", payload["metrics"])

    def test_sgd_update_reduces_loss_on_a_small_batch(self) -> None:
        datasets = build_partitioned_datasets(
            ("anna", "aria", "navi", "devin", "priya", "samira"), seed=3
        )
        model = initialize_context_mlp(datasets.train, seed=7)
        before, gradients = loss_and_gradients(datasets.train, model)
        updated = apply_sgd(model, gradients, learning_rate=0.1)
        after, _ = loss_and_gradients(datasets.train, updated)

        self.assertLess(after, before)

    def test_weight_decay_shrinks_weights_but_not_biases(self) -> None:
        datasets = build_partitioned_datasets(
            ("anna", "aria", "navi", "devin", "priya", "samira"), seed=3
        )
        model = initialize_context_mlp(datasets.train, seed=7)
        gradients = replace(
            model, **{name: values * 0 for name, values in model.__dict__.items()}
        )
        updated = apply_sgd(
            model, gradients, learning_rate=0.1, weight_decay=0.5
        )

        self.assertLess(abs(updated.input_weights).sum(), abs(model.input_weights).sum())
        self.assertEqual(abs(updated.input_bias).sum(), abs(model.input_bias).sum())

    def test_minibatch_training_applies_gradient_clipping(self) -> None:
        datasets = build_partitioned_datasets(
            tuple(f"name{letter}" for letter in "abcdefghijkl"), seed=3
        )
        base = TrainingConfig(
            epochs=2,
            batch_size=8,
            learning_rate=1.0,
            hidden_dim=16,
            seed=7,
        )
        unclipped = train_minibatch_context_mlp(datasets.train, base)
        clipped = train_minibatch_context_mlp(
            datasets.train, replace(base, max_gradient_norm=0.01)
        )

        self.assertNotEqual(
            model_fingerprint(unclipped.model), model_fingerprint(clipped.model)
        )
        self.assertTrue(all(loss > 0 for loss in clipped.training_nll))

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

    def test_early_stopping_halts_after_stale_development_epochs(self) -> None:
        datasets = build_partitioned_datasets(
            tuple(f"name{letter}" for letter in "abcdefghijkl"), seed=3
        )
        result = train_and_validate_context_mlp(
            datasets,
            TrainingConfig(
                epochs=20,
                batch_size=16,
                hidden_dim=16,
                seed=7,
                patience=2,
                minimum_delta=100.0,
            ),
        )

        self.assertTrue(result.stopped_early)
        self.assertEqual(len(result.development_nll), 3)
        self.assertEqual(result.best_epoch, 0)

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

    def test_partition_metrics_report_generalization_gaps(self) -> None:
        datasets = build_partitioned_datasets(
            tuple(f"name{letter}" for letter in "abcdefghijkl"), seed=3
        )
        result = train_and_validate_context_mlp(
            datasets,
            TrainingConfig(epochs=4, batch_size=16, hidden_dim=16, seed=7),
        )
        metrics = evaluate_partitions(datasets, result.best_model)

        self.assertAlmostEqual(
            metrics.development_gap,
            metrics.development.nll - metrics.train.nll,
        )
        self.assertAlmostEqual(metrics.test_gap, metrics.test.nll - metrics.train.nll)
        self.assertGreater(metrics.test.perplexity, 0)


if __name__ == "__main__":
    unittest.main()
