from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.attention import (  # noqa: E402
    AttentionConfig,
    ProjectionWeights,
    attention_invariants,
    causal_attention_mask,
    format_matrix,
    make_random_attention_problem,
    masked_softmax,
    max_reference_error,
    project_qkv,
    reference_attention_loop,
    render_attention_markdown,
    run_attention_experiment,
    scaled_dot_product_scores,
    single_head_attention,
    stable_softmax,
    validate_attention_trace,
    weighted_value_sum,
)


class AttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AttentionConfig(
            sequence_length=4, model_dim=6, head_dim=3, seed=13, causal=True
        )
        self.inputs, self.projections = make_random_attention_problem(self.config)
        self.trace = single_head_attention(self.inputs, self.projections, causal=True)

    def test_config_rejects_invalid_dimensions_seed_and_causal_flag(self) -> None:
        for keyword in ("sequence_length", "model_dim", "head_dim"):
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                AttentionConfig(**{keyword: 0})
        with self.assertRaises(ValueError):
            AttentionConfig(seed=True)
        with self.assertRaises(ValueError):
            AttentionConfig(causal=1)  # type: ignore[arg-type]

    def test_random_problem_is_reproducible(self) -> None:
        other_inputs, other_projections = make_random_attention_problem(self.config)
        np.testing.assert_array_equal(self.inputs, other_inputs)
        np.testing.assert_array_equal(self.projections.query, other_projections.query)
        np.testing.assert_array_equal(self.projections.key, other_projections.key)
        np.testing.assert_array_equal(self.projections.value, other_projections.value)

    def test_different_seed_changes_inputs(self) -> None:
        other_inputs, _ = make_random_attention_problem(replace(self.config, seed=14))
        self.assertFalse(np.array_equal(self.inputs, other_inputs))

    def test_qkv_projection_shapes(self) -> None:
        queries, keys, values = project_qkv(self.inputs, self.projections)
        self.assertEqual(queries.shape, (4, 3))
        self.assertEqual(keys.shape, (4, 3))
        self.assertEqual(values.shape, (4, 3))

    def test_projection_rejects_wrong_model_dimension(self) -> None:
        broken = ProjectionWeights(
            query=np.zeros((5, 3)),
            key=self.projections.key,
            value=self.projections.value,
        )
        with self.assertRaisesRegex(ValueError, "model_dim"):
            project_qkv(self.inputs, broken)

    def test_projection_rejects_mismatched_head_dimensions(self) -> None:
        broken = ProjectionWeights(
            query=self.projections.query,
            key=np.zeros((6, 2)),
            value=self.projections.value,
        )
        with self.assertRaisesRegex(ValueError, "share head_dim"):
            project_qkv(self.inputs, broken)

    def test_scaled_scores_match_hand_calculation(self) -> None:
        queries = np.array([[1.0, 0.0], [0.0, 2.0]])
        keys = np.array([[1.0, 1.0], [2.0, 0.0]])
        scores, scale = scaled_dot_product_scores(queries, keys)
        self.assertAlmostEqual(scale, 1.0 / np.sqrt(2.0))
        np.testing.assert_allclose(
            scores,
            np.array([[1.0, 2.0], [2.0, 0.0]]) / np.sqrt(2.0),
        )

    def test_scaled_scores_reject_head_dimension_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "share head_dim"):
            scaled_dot_product_scores(np.ones((2, 3)), np.ones((2, 4)))

    def test_stable_softmax_handles_large_logits(self) -> None:
        probabilities = stable_softmax(np.array([1_000.0, 1_001.0]))
        expected = np.array([1.0, np.e]) / (1.0 + np.e)
        np.testing.assert_allclose(probabilities, expected)

    def test_stable_softmax_rejects_non_finite_values_and_bad_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            stable_softmax([0.0, np.inf])
        with self.assertRaisesRegex(ValueError, "axis"):
            stable_softmax([0.0, 1.0], axis=2)

    def test_square_causal_mask_is_lower_triangular(self) -> None:
        expected = np.array(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        np.testing.assert_array_equal(causal_attention_mask(4), expected)

    def test_rectangular_causal_mask_includes_prefix_keys(self) -> None:
        expected = np.array([[True, True, True, False], [True, True, True, True]])
        np.testing.assert_array_equal(causal_attention_mask(2, 4), expected)

    def test_causal_mask_rejects_short_key_sequence(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least"):
            causal_attention_mask(3, 2)

    def test_masked_softmax_uses_exact_zeros(self) -> None:
        scores = np.array([[1.0, 1000.0], [2.0, 3.0]])
        allowed = np.array([[True, False], [True, True]])
        probabilities = masked_softmax(scores, allowed)
        self.assertEqual(probabilities[0, 1], 0.0)
        np.testing.assert_allclose(probabilities.sum(axis=-1), np.ones(2))

    def test_masked_softmax_rejects_fully_masked_row(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            masked_softmax(np.ones((2, 2)), np.array([[True, False], [False, False]]))

    def test_masked_softmax_rejects_non_boolean_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "booleans"):
            masked_softmax(np.ones((2, 2)), np.ones((2, 2)))

    def test_causal_attention_has_normalized_rows_and_no_future_weight(self) -> None:
        np.testing.assert_allclose(self.trace.weights.sum(axis=-1), np.ones(4))
        self.assertTrue(np.all(self.trace.weights[np.triu_indices(4, 1)] == 0.0))

    def test_first_causal_token_copies_its_only_available_value(self) -> None:
        np.testing.assert_allclose(self.trace.context[0], self.trace.values[0])

    def test_noncausal_attention_allows_every_key(self) -> None:
        trace = single_head_attention(self.inputs, self.projections, causal=False)
        self.assertTrue(trace.allowed.all())
        self.assertTrue((trace.weights > 0.0).all())

    def test_uniform_scores_average_the_values(self) -> None:
        inputs = np.array([[1.0], [3.0]])
        projections = ProjectionWeights(
            query=np.zeros((1, 1)),
            key=np.ones((1, 1)),
            value=np.ones((1, 1)),
        )
        trace = single_head_attention(inputs, projections, causal=False)
        np.testing.assert_allclose(trace.weights, np.full((2, 2), 0.5))
        np.testing.assert_allclose(trace.context, np.full((2, 1), 2.0))

    def test_weighted_value_sum_rejects_unnormalized_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            weighted_value_sum(np.ones((2, 2)), np.ones((2, 1)))

    def test_vectorized_attention_matches_independent_loop(self) -> None:
        reference = reference_attention_loop(self.inputs, self.projections, causal=True)
        np.testing.assert_allclose(self.trace.context, reference, atol=1e-12, rtol=0.0)
        self.assertLessEqual(
            max_reference_error(self.trace, self.inputs, self.projections, causal=True),
            1e-12,
        )

    def test_trace_validation_reports_all_invariants(self) -> None:
        metrics = validate_attention_trace(self.trace)
        self.assertTrue(metrics["shape_ok"])
        self.assertTrue(metrics["all_finite"])
        self.assertLessEqual(float(metrics["max_row_sum_error"]), 1e-12)
        self.assertEqual(metrics["max_masked_weight"], 0.0)

    def test_trace_validation_detects_masked_probability(self) -> None:
        broken_weights = self.trace.weights.copy()
        broken_weights[0, 1] = 0.1
        broken = replace(self.trace, weights=broken_weights)
        metrics = attention_invariants(broken)
        self.assertEqual(metrics["max_masked_weight"], 0.1)
        with self.assertRaisesRegex(ValueError, "sum to one|Masked|masked"):
            validate_attention_trace(broken)

    def test_trace_validation_reports_shape_error_for_malformed_trace(self) -> None:
        broken = replace(self.trace, weights=self.trace.weights[:, :-1])
        metrics = attention_invariants(broken)
        self.assertFalse(metrics["shape_ok"])
        with self.assertRaisesRegex(ValueError, "tensor shape"):
            validate_attention_trace(broken)

    def test_experiment_and_markdown_are_deterministic(self) -> None:
        first = run_attention_experiment(self.config)
        second = run_attention_experiment(self.config)
        np.testing.assert_array_equal(first.context, second.context)
        metrics = validate_attention_trace(first)
        markdown = render_attention_markdown(
            self.config,
            first,
            metrics,
            max_reference_error(first, self.inputs, self.projections, causal=True),
        )
        self.assertIn("`(T, C) @ (C, D) -> (T, D)`", markdown)
        self.assertIn("maximum masked weight: `0.000e+00`", markdown)

    def test_matrix_formatter_validates_precision(self) -> None:
        self.assertEqual(format_matrix([[1.25]], precision=2), "[[1.25]]")
        with self.assertRaisesRegex(ValueError, "precision"):
            format_matrix([[1.0]], precision=-1)


if __name__ == "__main__":
    unittest.main()
