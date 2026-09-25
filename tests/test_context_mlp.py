from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.context_mlp import (
    ContextMLPError,
    build_context_dataset,
    build_split_datasets,
    create_minibatches,
    initialize_context_mlp,
    loss_and_gradients,
    predict_probabilities,
    split_records,
    train_context_mlp,
)


class ContextDatasetTests(unittest.TestCase):
    def test_record_split_is_deterministic_and_disjoint(self) -> None:
        words = ("anna", "aria", "navi", "devin", "priya")
        first = split_records(words, validation_fraction=0.4, seed=7)
        second = split_records(words, validation_fraction=0.4, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first.validation), 2)
        self.assertFalse(set(first.train) & set(first.validation))
        self.assertEqual(set(first.train) | set(first.validation), set(words))

    def test_record_split_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ContextMLPError):
            split_records(("anna",))
        with self.assertRaises(ContextMLPError):
            split_records(("anna", "aria"), validation_fraction=1.0)

    def test_split_datasets_share_one_vocabulary(self) -> None:
        datasets = build_split_datasets(
            ("anna", "aria", "navi", "devin", "priya"), seed=3
        )
        self.assertEqual(
            datasets.train.vocabulary.tokens,
            datasets.validation.vocabulary.tokens,
        )
        self.assertIn("p", datasets.train.vocabulary.tokens)

    def test_contexts_are_boundary_aware(self) -> None:
        dataset = build_context_dataset(("ab",), block_size=3)
        decoded = [
            (
                tuple(dataset.vocabulary.decode(int(token)) for token in context),
                dataset.vocabulary.decode(int(target)),
            )
            for context, target in zip(dataset.contexts, dataset.targets)
        ]
        self.assertEqual(
            decoded,
            [
                ((".", ".", "."), "a"),
                ((".", ".", "a"), "b"),
                ((".", "a", "b"), "."),
            ],
        )

    def test_each_record_starts_with_a_fresh_context(self) -> None:
        dataset = build_context_dataset(("ab", "cd"), block_size=2)
        boundary = dataset.vocabulary.encode(".")
        np.testing.assert_array_equal(dataset.contexts[3], [boundary, boundary])

    def test_sample_count_includes_one_end_target_per_record(self) -> None:
        dataset = build_context_dataset(("ab", "cde"), block_size=2)
        self.assertEqual(dataset.sample_count, 7)
        self.assertEqual(dataset.contexts.shape, (7, 2))

    def test_invalid_block_size_is_rejected(self) -> None:
        for value in (True, 0, -1):
            with (
                self.subTest(value=value),
                self.assertRaises((TypeError, ContextMLPError)),
            ):
                build_context_dataset(("ab",), block_size=value)

    def test_minibatches_are_deterministic_and_cover_each_sample(self) -> None:
        dataset = build_context_dataset(("anna", "aria", "navi"))
        first = create_minibatches(dataset, batch_size=4, seed=17)
        second = create_minibatches(dataset, batch_size=4, seed=17)
        self.assertEqual([batch.sample_count for batch in first], [4, 4, 4, 3])
        np.testing.assert_array_equal(
            np.concatenate([batch.targets for batch in first]),
            np.concatenate([batch.targets for batch in second]),
        )
        observed = sorted(
            zip(
                np.concatenate([batch.contexts for batch in first]).tolist(),
                np.concatenate([batch.targets for batch in first]).tolist(),
            )
        )
        expected = sorted(zip(dataset.contexts.tolist(), dataset.targets.tolist()))
        self.assertEqual(observed, expected)


class ContextMLPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = build_context_dataset(("anna", "aria", "navi"))

    def test_initialization_has_expected_shapes(self) -> None:
        model = initialize_context_mlp(
            self.dataset, embedding_dim=4, hidden_dim=12, seed=7
        )
        self.assertEqual(model.embeddings.shape, (self.dataset.vocabulary.size, 4))
        self.assertEqual(model.input_weights.shape, (12, 12))
        self.assertEqual(model.output_weights.shape, (12, self.dataset.vocabulary.size))

    def test_initialization_is_seeded(self) -> None:
        first = initialize_context_mlp(self.dataset, seed=11)
        second = initialize_context_mlp(self.dataset, seed=11)
        for name in first.__dict__:
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    def test_probabilities_are_normalized(self) -> None:
        model = initialize_context_mlp(self.dataset, seed=3)
        probabilities = predict_probabilities(self.dataset, model)
        self.assertEqual(
            probabilities.shape,
            (self.dataset.sample_count, self.dataset.vocabulary.size),
        )
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.all(probabilities > 0))

    def test_shape_mismatch_is_rejected(self) -> None:
        model = initialize_context_mlp(self.dataset, seed=3)
        with self.assertRaises(ContextMLPError):
            predict_probabilities(
                self.dataset,
                replace(model, input_weights=model.input_weights[:-1]),
            )

    def test_all_gradients_match_finite_differences(self) -> None:
        model = initialize_context_mlp(
            self.dataset, embedding_dim=3, hidden_dim=5, seed=5
        )
        _, gradients = loss_and_gradients(self.dataset, model)
        epsilon = 1e-6
        probes = {
            "embeddings": (1, 1),
            "input_weights": (0, 0),
            "input_bias": (0,),
            "output_weights": (0, 2),
            "output_bias": (2,),
        }

        for name, index in probes.items():
            with self.subTest(parameter=name):
                positive_values = getattr(model, name).copy()
                negative_values = getattr(model, name).copy()
                positive_values[index] += epsilon
                negative_values[index] -= epsilon
                positive, _ = loss_and_gradients(
                    self.dataset, replace(model, **{name: positive_values})
                )
                negative, _ = loss_and_gradients(
                    self.dataset, replace(model, **{name: negative_values})
                )
                numeric = (positive - negative) / (2 * epsilon)
                self.assertAlmostEqual(
                    getattr(gradients, name)[index], numeric, places=7
                )

    def test_training_reduces_loss(self) -> None:
        result = train_context_mlp(
            self.dataset,
            embedding_dim=4,
            hidden_dim=16,
            steps=100,
            learning_rate=0.2,
            seed=9,
        )
        self.assertLess(result.losses[-1], result.losses[0])

    def test_training_is_deterministic(self) -> None:
        first = train_context_mlp(self.dataset, steps=10, seed=13)
        second = train_context_mlp(self.dataset, steps=10, seed=13)
        self.assertEqual(first.losses, second.losses)
        for name in first.model.__dict__:
            np.testing.assert_array_equal(
                getattr(first.model, name), getattr(second.model, name)
            )


if __name__ == "__main__":
    unittest.main()
