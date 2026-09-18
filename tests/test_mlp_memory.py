from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.mlp_memory import (  # noqa: E402
    MLPWeights,
    change_basis,
    contribution_norms,
    cosine_similarity,
    direct_mlp_outputs,
    experiment_metrics,
    feature_gram_matrix,
    make_fact_memory,
    max_feature_coherence,
    max_reconstruction_error,
    mean_squared_error,
    mlp_forward,
    reconstruct_outputs,
    regular_feature_directions,
    relu,
    render_memory_markdown,
    run_memory_experiment,
    superposition_reconstruct,
    top_contributing_neurons,
    validate_mlp_weights,
)


class MLPMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queries, self.weights = make_fact_memory()
        self.trace = mlp_forward(self.queries, self.weights)

    def test_relu_clips_negative_values_and_preserves_positive_values(self) -> None:
        np.testing.assert_array_equal(relu([-2.0, 0.0, 3.0]), [0.0, 0.0, 3.0])

    def test_relu_rejects_empty_and_non_finite_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            relu([])
        with self.assertRaisesRegex(ValueError, "finite"):
            relu([np.inf])

    def test_fact_memory_parameter_shapes_are_valid(self) -> None:
        self.assertEqual(validate_mlp_weights(self.weights), (4, 4, 3))

    def test_weight_validation_rejects_wrong_key_bias_shape(self) -> None:
        broken = replace(self.weights, key_bias=np.zeros(3))
        with self.assertRaisesRegex(ValueError, "key_bias"):
            validate_mlp_weights(broken)

    def test_weight_validation_rejects_hidden_width_mismatch(self) -> None:
        broken = replace(self.weights, values=np.zeros((3, 3)))
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            validate_mlp_weights(broken)

    def test_weight_validation_rejects_output_bias_mismatch(self) -> None:
        broken = replace(self.weights, output_bias=np.zeros(2))
        with self.assertRaisesRegex(ValueError, "output_bias"):
            validate_mlp_weights(broken)

    def test_forward_exposes_expected_shapes(self) -> None:
        self.assertEqual(self.trace.preactivations.shape, (4, 4))
        self.assertEqual(self.trace.activations.shape, (4, 4))
        self.assertEqual(self.trace.contributions.shape, (4, 4, 3))
        self.assertEqual(self.trace.outputs.shape, (4, 3))

    def test_forward_rejects_wrong_input_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_dim"):
            mlp_forward(np.zeros((2, 3)), self.weights)

    def test_each_fact_query_activates_one_memory(self) -> None:
        expected = 0.5 * np.eye(4)
        np.testing.assert_allclose(self.trace.activations, expected)
        np.testing.assert_array_equal(
            top_contributing_neurons(self.trace).reshape(-1), np.arange(4)
        )

    def test_output_is_sum_of_per_neuron_contributions(self) -> None:
        reconstructed = reconstruct_outputs(self.trace, self.weights.output_bias)
        np.testing.assert_allclose(reconstructed, self.trace.outputs)
        self.assertEqual(
            max_reconstruction_error(self.trace, self.weights.output_bias), 0.0
        )

    def test_expanded_output_matches_direct_matrix_expression(self) -> None:
        direct = direct_mlp_outputs(self.queries, self.weights)
        np.testing.assert_allclose(self.trace.outputs, direct)

    def test_contribution_norms_match_activation_scaled_value_norms(self) -> None:
        expected = self.trace.activations * np.linalg.norm(self.weights.values, axis=1)
        np.testing.assert_allclose(contribution_norms(self.trace), expected)

    def test_top_contributor_can_return_multiple_stable_indexes(self) -> None:
        indexes = top_contributing_neurons(self.trace, count=2)
        self.assertEqual(indexes.shape, (4, 2))
        np.testing.assert_array_equal(indexes[:, 0], np.arange(4))

    def test_top_contributor_rejects_bad_counts(self) -> None:
        for count in (0, True, 5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                top_contributing_neurons(self.trace, count=count)  # type: ignore[arg-type]

    def test_cosine_similarity_handles_parallel_and_orthogonal_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [2.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 2.0]), 0.0)

    def test_cosine_similarity_rejects_zero_and_mismatched_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero"):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "same shape"):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_change_of_basis_preserves_preactivations_and_outputs(self) -> None:
        basis = np.array(
            [
                [1.0, 0.2, 0.0, 0.0],
                [0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 1.0, 0.4],
                [0.1, 0.0, 0.0, 1.0],
            ]
        )
        transformed_inputs, transformed_weights = change_basis(
            self.queries, self.weights, basis
        )
        transformed_trace = mlp_forward(transformed_inputs, transformed_weights)
        np.testing.assert_allclose(
            transformed_trace.preactivations, self.trace.preactivations
        )
        np.testing.assert_allclose(transformed_trace.outputs, self.trace.outputs)

    def test_change_of_basis_rejects_singular_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "invertible"):
            change_basis(self.queries, self.weights, np.zeros((4, 4)))

    def test_regular_feature_directions_are_unit_length(self) -> None:
        directions = regular_feature_directions(5)
        self.assertEqual(directions.shape, (2, 5))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=0), np.ones(5))

    def test_regular_feature_directions_validate_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            regular_feature_directions(0)
        with self.assertRaisesRegex(ValueError, "requires"):
            regular_feature_directions(5, representation_dim=3)

    def test_feature_gram_matrix_is_symmetric_with_unit_diagonal(self) -> None:
        gram = feature_gram_matrix(regular_feature_directions(5))
        np.testing.assert_allclose(gram, gram.T)
        np.testing.assert_allclose(np.diag(gram), np.ones(5))

    def test_feature_coherence_exposes_non_orthogonal_directions(self) -> None:
        coherence = max_feature_coherence(regular_feature_directions(5))
        self.assertGreater(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
        self.assertEqual(max_feature_coherence(np.array([[1.0], [0.0]])), 0.0)

    def test_superposition_reconstruction_has_expected_shape_and_nonnegative_values(
        self,
    ) -> None:
        features = np.eye(5)
        reconstructed = superposition_reconstruct(
            features, regular_feature_directions(5)
        )
        self.assertEqual(reconstructed.shape, features.shape)
        self.assertTrue(np.all(reconstructed >= 0.0))

    def test_superposition_reconstruction_validates_width_and_bias(self) -> None:
        directions = regular_feature_directions(5)
        with self.assertRaisesRegex(ValueError, "feature width"):
            superposition_reconstruct(np.eye(4), directions)
        with self.assertRaisesRegex(ValueError, "bias"):
            superposition_reconstruct(np.eye(5), directions, bias=-1.0)

    def test_mean_squared_error_matches_hand_calculation(self) -> None:
        self.assertAlmostEqual(mean_squared_error([1.0, 3.0], [1.0, 1.0]), 2.0)
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            mean_squared_error([1.0], [1.0, 2.0])

    def test_collision_increases_superposition_error(self) -> None:
        experiment = run_memory_experiment()
        metrics = experiment_metrics(experiment)
        self.assertGreater(
            float(metrics["collision_reconstruction_mse"]),
            float(metrics["sparse_reconstruction_mse"]),
        )
        self.assertTrue(metrics["collision_increases_error"])

    def test_experiment_is_deterministic_and_valid(self) -> None:
        first = run_memory_experiment()
        second = run_memory_experiment()
        np.testing.assert_array_equal(first.trace.outputs, second.trace.outputs)
        metrics = experiment_metrics(first)
        self.assertTrue(metrics["key_value_expansion_exact"])
        self.assertLessEqual(float(metrics["direct_max_error"]), 1e-12)
        self.assertLessEqual(float(metrics["basis_invariance_max_error"]), 1e-12)

    def test_markdown_reports_scope_and_numerical_checks(self) -> None:
        experiment = run_memory_experiment()
        markdown = render_memory_markdown(experiment, experiment_metrics(experiment))
        self.assertIn("direct-versus-expanded maximum error: `0.000e+00`", markdown)
        self.assertIn("changed-basis output maximum error", markdown)
        self.assertIn("facts live in one neuron", markdown)

    def test_output_reconstruction_rejects_wrong_bias_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "contribution width"):
            reconstruct_outputs(self.trace, np.zeros(2))

    def test_direct_expression_rejects_non_finite_weights(self) -> None:
        broken = MLPWeights(
            keys=np.full((4, 4), np.nan),
            key_bias=self.weights.key_bias,
            values=self.weights.values,
            output_bias=self.weights.output_bias,
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            direct_mlp_outputs(self.queries, broken)


if __name__ == "__main__":
    unittest.main()
