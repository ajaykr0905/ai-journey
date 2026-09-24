from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.bigram_lm import DEFAULT_CORPUS, build_bigram_model
from ai_journey.neural_bigram import (
    NeuralBigramValidationError,
    additive_smoothed_probabilities,
    build_bigram_dataset,
    count_model_loss,
    experiment_metrics,
    experiment_payload,
    finite_difference_probes,
    log_count_weights,
    logits_from_weights,
    loss_and_gradient,
    negative_log_likelihood,
    neural_bigram_loss,
    one_hot,
    perplexity,
    probability_model_from_weights,
    render_dataflow_mermaid,
    render_neural_bigram_markdown,
    run_neural_bigram_experiment,
    stable_softmax,
    train_neural_bigram,
    validate_dataset,
    validate_experiment,
    validate_probability_matrix,
    validate_smoothing,
    validate_weights,
)


class DatasetAndEncodingTests(unittest.TestCase):
    def test_dataset_encodes_boundary_aware_pairs_in_order(self) -> None:
        dataset = build_bigram_dataset(("ab",))
        decoded = tuple(
            (
                dataset.vocabulary.decode(int(previous)),
                dataset.vocabulary.decode(int(next_token)),
            )
            for previous, next_token in zip(dataset.inputs, dataset.targets)
        )
        self.assertEqual(decoded, ((".", "a"), ("a", "b"), ("b", ".")))

    def test_default_dataset_reconciles_to_day_19_transition_total(self) -> None:
        dataset = build_bigram_dataset(DEFAULT_CORPUS)
        self.assertEqual(dataset.sample_count, 120)
        self.assertEqual(dataset.vocabulary.size, 18)
        self.assertEqual(dataset.inputs.dtype, np.int64)

    def test_validate_dataset_rejects_misaligned_noninteger_and_out_of_range(
        self,
    ) -> None:
        dataset = build_bigram_dataset(("ada",))
        invalid = (
            replace(dataset, targets=dataset.targets[:-1]),
            replace(dataset, inputs=dataset.inputs.astype(np.float64)),
            replace(dataset, targets=np.full_like(dataset.targets, 999)),
        )
        for candidate in invalid:
            with (
                self.subTest(candidate=candidate),
                self.assertRaises((NeuralBigramValidationError, TypeError)),
            ):
                validate_dataset(candidate)

    def test_one_hot_has_one_active_value_per_row(self) -> None:
        ids = np.array([2, 0, 1, 2], dtype=np.int64)
        encoded = one_hot(ids, 3)
        self.assertEqual(encoded.shape, (4, 3))
        np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(4))
        np.testing.assert_array_equal(np.argmax(encoded, axis=1), ids)

    def test_one_hot_rejects_invalid_ids_shape_dtype_and_size(self) -> None:
        with self.assertRaises(NeuralBigramValidationError):
            one_hot(np.array([3], dtype=np.int64), 3)
        with self.assertRaises(TypeError):
            one_hot(np.array([[0]], dtype=np.int64), 3)
        with self.assertRaises(TypeError):
            one_hot(np.array([0.0]), 3)
        with self.assertRaises(NeuralBigramValidationError):
            one_hot(np.array([0], dtype=np.int64), 0)


class ProbabilityAndLossTests(unittest.TestCase):
    def test_logits_match_direct_weight_row_lookup(self) -> None:
        dataset = build_bigram_dataset(("ab",))
        size = dataset.vocabulary.size
        weights = np.arange(size * size, dtype=np.float64).reshape(size, size)
        np.testing.assert_array_equal(
            logits_from_weights(dataset, weights), weights[dataset.inputs]
        )

    def test_validate_weights_rejects_shape_integer_and_nonfinite_values(self) -> None:
        with self.assertRaises(NeuralBigramValidationError):
            validate_weights(np.zeros((2, 3), dtype=np.float64), 2)
        with self.assertRaises(TypeError):
            validate_weights(np.zeros((2, 2), dtype=np.int64), 2)
        with self.assertRaises(NeuralBigramValidationError):
            validate_weights(np.array([[0.0, np.nan], [0.0, 0.0]]), 2)

    def test_stable_softmax_handles_large_logits_and_shift_invariance(self) -> None:
        logits = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -999.0, -998.0]])
        probabilities = stable_softmax(logits)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_allclose(probabilities[0], probabilities[1])
        np.testing.assert_allclose(
            stable_softmax(logits + np.array([[7.0], [-31.0]])), probabilities
        )

    def test_stable_softmax_rejects_wrong_shape_and_nonfinite_values(self) -> None:
        with self.assertRaises(NeuralBigramValidationError):
            stable_softmax(np.array([1.0, 2.0]))
        with self.assertRaises(NeuralBigramValidationError):
            stable_softmax(np.array([[0.0, np.inf]]))

    def test_probability_validator_requires_positive_normalized_rows(self) -> None:
        validate_probability_matrix(np.array([[0.25, 0.75], [0.5, 0.5]]))
        for invalid in (
            np.array([[0.0, 1.0]]),
            np.array([[0.2, 0.2]]),
            np.array([[np.nan, 0.5]]),
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises(NeuralBigramValidationError),
            ):
                validate_probability_matrix(invalid)

    def test_negative_log_likelihood_matches_hand_calculation(self) -> None:
        probabilities = np.array([[0.8, 0.2], [0.25, 0.75]])
        targets = np.array([0, 1], dtype=np.int64)
        expected = -float(np.log([0.8, 0.75]).mean())
        self.assertAlmostEqual(
            negative_log_likelihood(probabilities, targets), expected
        )

    def test_negative_log_likelihood_rejects_bad_targets(self) -> None:
        probabilities = np.array([[0.4, 0.6]])
        with self.assertRaises(NeuralBigramValidationError):
            negative_log_likelihood(probabilities, np.array([0, 1], dtype=np.int64))
        with self.assertRaises(NeuralBigramValidationError):
            negative_log_likelihood(probabilities, np.array([2], dtype=np.int64))
        with self.assertRaises(TypeError):
            negative_log_likelihood(probabilities, np.array([0.0]))

    def test_perplexity_is_exponential_of_nll_and_validated(self) -> None:
        self.assertAlmostEqual(perplexity(float(np.log(3.0))), 3.0)
        for invalid in (-1.0, float("nan"), float("inf"), 1000.0):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises(NeuralBigramValidationError),
            ):
                perplexity(invalid)


class SmoothingAndEquivalenceTests(unittest.TestCase):
    def test_smoothing_requires_a_positive_finite_real(self) -> None:
        self.assertEqual(validate_smoothing(1), 1.0)
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises(NeuralBigramValidationError),
            ):
                validate_smoothing(invalid)
        with self.assertRaises(TypeError):
            validate_smoothing(True)

    def test_additive_smoothing_makes_every_probability_positive(self) -> None:
        counts = np.array([[2, 0], [0, 0]], dtype=np.int64)
        probabilities = additive_smoothed_probabilities(counts, 1.0)
        np.testing.assert_allclose(probabilities, [[0.75, 0.25], [0.5, 0.5]])
        self.assertTrue(np.all(probabilities > 0.0))

    def test_additive_smoothing_rejects_bad_counts(self) -> None:
        with self.assertRaises(NeuralBigramValidationError):
            additive_smoothed_probabilities(np.ones((2, 3), dtype=np.int64))
        with self.assertRaises(TypeError):
            additive_smoothed_probabilities(np.ones((2, 2), dtype=np.float64))
        with self.assertRaises(NeuralBigramValidationError):
            additive_smoothed_probabilities(np.array([[1, -1], [0, 1]]))

    def test_log_count_weights_softmax_exactly_matches_smoothed_counts(self) -> None:
        model = build_bigram_model(DEFAULT_CORPUS)
        smoothed = additive_smoothed_probabilities(model.counts, 1.0)
        probabilities = stable_softmax(log_count_weights(model.counts, 1.0))
        np.testing.assert_allclose(probabilities, smoothed, atol=1e-15)

    def test_count_and_neural_losses_match_for_log_count_weights(self) -> None:
        dataset = build_bigram_dataset(DEFAULT_CORPUS)
        model = build_bigram_model(DEFAULT_CORPUS)
        smoothed = additive_smoothed_probabilities(model.counts, 1.0)
        weights = log_count_weights(model.counts, 1.0)
        self.assertAlmostEqual(
            count_model_loss(dataset, smoothed),
            neural_bigram_loss(dataset, weights),
            places=14,
        )

    def test_count_loss_rejects_wrong_vocabulary_shape(self) -> None:
        dataset = build_bigram_dataset(("ab",))
        with self.assertRaises(NeuralBigramValidationError):
            count_model_loss(dataset, np.full((2, 2), 0.5))

    def test_probability_model_from_weights_is_sampling_compatible(self) -> None:
        model = build_bigram_model(DEFAULT_CORPUS)
        neural_model = probability_model_from_weights(
            model, log_count_weights(model.counts, 1.0)
        )
        np.testing.assert_allclose(
            neural_model.probabilities,
            additive_smoothed_probabilities(model.counts, 1.0),
        )


class GradientAndTrainingTests(unittest.TestCase):
    def test_analytic_gradient_shape_is_finite_and_balanced_by_row(self) -> None:
        dataset = build_bigram_dataset(DEFAULT_CORPUS)
        weights = np.zeros((18, 18), dtype=np.float64)
        loss, gradient = loss_and_gradient(dataset, weights)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(gradient.shape, weights.shape)
        self.assertTrue(np.all(np.isfinite(gradient)))
        np.testing.assert_allclose(gradient.sum(axis=1), 0.0, atol=1e-15)

    def test_centered_difference_probes_match_analytic_gradient(self) -> None:
        dataset = build_bigram_dataset(("ada", "ava"))
        size = dataset.vocabulary.size
        weights = np.linspace(-0.2, 0.2, size * size).reshape(size, size)
        probes = finite_difference_probes(dataset, weights, ((0, 1), (1, 2), (3, 0)))
        self.assertEqual(len(probes), 3)
        self.assertLess(max(probe.absolute_error for probe in probes), 1e-9)

    def test_gradient_probe_validation_rejects_empty_duplicate_and_out_of_range(
        self,
    ) -> None:
        dataset = build_bigram_dataset(("ada",))
        weights = np.zeros((3, 3), dtype=np.float64)
        with self.assertRaises(NeuralBigramValidationError):
            finite_difference_probes(dataset, weights, ())
        with self.assertRaises(NeuralBigramValidationError):
            finite_difference_probes(dataset, weights, ((0, 0), (0, 0)))
        with self.assertRaises(NeuralBigramValidationError):
            finite_difference_probes(dataset, weights, ((9, 0),))

    def test_training_is_deterministic_and_reduces_loss(self) -> None:
        dataset = build_bigram_dataset(DEFAULT_CORPUS)
        first_weights, first_trace = train_neural_bigram(dataset, steps=50)
        second_weights, second_trace = train_neural_bigram(dataset, steps=50)
        np.testing.assert_array_equal(first_weights, second_weights)
        self.assertEqual(first_trace, second_trace)
        self.assertLess(first_trace[-1].loss, first_trace[0].loss)

    def test_training_trace_includes_first_last_and_intervals(self) -> None:
        dataset = build_bigram_dataset(("ada", "ava"))
        _, trace = train_neural_bigram(
            dataset, steps=12, learning_rate=2.0, checkpoint_interval=5
        )
        self.assertEqual([checkpoint.step for checkpoint in trace], [0, 5, 10, 12])

    def test_training_rejects_invalid_hyperparameters(self) -> None:
        dataset = build_bigram_dataset(("ada",))
        with self.assertRaises(NeuralBigramValidationError):
            train_neural_bigram(dataset, steps=0)
        with self.assertRaises(NeuralBigramValidationError):
            train_neural_bigram(dataset, learning_rate=0.0)
        with self.assertRaises(TypeError):
            train_neural_bigram(dataset, checkpoint_interval=True)


class ExperimentAndReportTests(unittest.TestCase):
    def test_default_experiment_matches_losses_checks_gradients_and_trains(
        self,
    ) -> None:
        experiment = run_neural_bigram_experiment()
        metrics = experiment_metrics(experiment)
        self.assertEqual(metrics["examples"], 120)
        self.assertEqual(metrics["one_hot_shape"], "120x18")
        self.assertEqual(metrics["weight_shape"], "18x18")
        self.assertLess(metrics["equivalence_error"], 1e-12)
        self.assertLess(metrics["max_gradient_error"], 1e-8)
        self.assertGreater(metrics["loss_reduction_percent"], 0.0)

    def test_experiment_is_deterministic_for_same_configuration(self) -> None:
        first = run_neural_bigram_experiment(steps=30, seed=20)
        second = run_neural_bigram_experiment(steps=30, seed=20)
        np.testing.assert_array_equal(first.trained_weights, second.trained_weights)
        self.assertEqual(first.training_trace, second.training_trace)
        self.assertEqual(first.samples, second.samples)

    def test_experiment_payload_is_json_serializable_and_complete(self) -> None:
        payload = experiment_payload(
            run_neural_bigram_experiment(steps=20, sample_count=3)
        )
        encoded = json.dumps(payload)
        self.assertIn("trained_weights", payload)
        self.assertEqual(len(payload["gradient_probes"]), 6)  # type: ignore[arg-type]
        self.assertIn("equivalence_error", encoded)

    def test_validate_experiment_detects_loss_gradient_and_training_tampering(
        self,
    ) -> None:
        experiment = run_neural_bigram_experiment(steps=20)
        with self.assertRaises(NeuralBigramValidationError):
            validate_experiment(replace(experiment, count_loss=99.0))
        bad_probe = replace(experiment.gradient_probes[0], absolute_error=1.0)
        with self.assertRaises(NeuralBigramValidationError):
            validate_experiment(
                replace(
                    experiment,
                    gradient_probes=(bad_probe, *experiment.gradient_probes[1:]),
                )
            )
        with self.assertRaises(NeuralBigramValidationError):
            validate_experiment(
                replace(experiment, trained_loss=experiment.initial_loss)
            )

    def test_markdown_contains_shapes_equivalence_training_and_disclaimer(self) -> None:
        report = render_neural_bigram_markdown(
            run_neural_bigram_experiment(steps=20, sample_count=2)
        )
        self.assertIn("One-hot matrix: `120x18`", report)
        self.assertIn("Count-to-neural equivalence", report)
        self.assertIn("Absolute loss difference", report)
        self.assertIn("Maximum gradient error", report)
        self.assertIn("does not prove", report)
        self.assertIn("```mermaid", report)

    def test_mermaid_dataflow_labels_every_major_tensor_shape(self) -> None:
        graph = render_dataflow_mermaid(
            run_neural_bigram_experiment(steps=10, sample_count=1)
        )
        for shape in ("(120,)", "(120, 18)", "(18, 18)"):
            self.assertIn(shape, graph)
        self.assertIn("negative log-likelihood", graph)


if __name__ == "__main__":
    unittest.main()
