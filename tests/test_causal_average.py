from __future__ import annotations

import unittest

import torch

from ai_journey.causal_average import (
    CausalAverageAudit,
    CausalAverageError,
    audit_equivalence,
    audit_weights,
    causal_average_cumsum,
    causal_average_loop,
    causal_average_matmul,
    causal_average_softmax,
    causal_mask,
    masked_softmax_weights,
    require_equivalence,
    triangular_average_weights,
    validate_values,
)


class CausalInputTests(unittest.TestCase):
    def test_accepts_single_and_batched_float_sequences(self) -> None:
        validate_values(torch.ones(4, 3, dtype=torch.float64))
        validate_values(torch.ones(2, 4, 3, dtype=torch.float32))

    def test_rejects_wrong_rank_empty_integer_and_nonfinite_inputs(self) -> None:
        for values, message in (
            (torch.ones(4), "shape"),
            (torch.ones(0, 3), "non-empty"),
            (torch.ones(4, 3, dtype=torch.long), "floating-point"),
            (torch.tensor([[float("nan")]]), "finite"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(CausalAverageError, message):
                    validate_values(values)
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            validate_values([[1.0]])  # type: ignore[arg-type]

    def test_builds_exact_lower_triangular_mask(self) -> None:
        expected = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        self.assertTrue(torch.equal(causal_mask(4), expected))

    def test_rejects_invalid_mask_lengths(self) -> None:
        with self.assertRaisesRegex(TypeError, "integer"):
            causal_mask(True)
        with self.assertRaisesRegex(CausalAverageError, "positive"):
            causal_mask(0)


class LoopAverageTests(unittest.TestCase):
    def test_computes_each_prefix_mean_by_hand(self) -> None:
        values = torch.tensor([[2.0, 4.0], [4.0, 8.0], [9.0, 3.0]])
        expected = torch.tensor([[2.0, 4.0], [3.0, 6.0], [5.0, 5.0]])
        torch.testing.assert_close(causal_average_loop(values), expected)

    def test_preserves_batch_channel_dtype_and_gradient_flow(self) -> None:
        values = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3)
        values.requires_grad_()
        output = causal_average_loop(values)
        self.assertEqual(output.shape, values.shape)
        self.assertEqual(output.dtype, torch.float64)
        output.square().sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())


class TriangularWeightTests(unittest.TestCase):
    def test_normalizes_each_allowed_prefix_uniformly(self) -> None:
        weights = triangular_average_weights(4, dtype=torch.float64)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones(4, dtype=torch.float64)
        )

    def test_rejects_nonfloating_weight_dtype(self) -> None:
        with self.assertRaisesRegex(CausalAverageError, "floating-point"):
            triangular_average_weights(3, dtype=torch.long)


class MatrixAverageTests(unittest.TestCase):
    def test_matches_the_loop_oracle_for_unbatched_values(self) -> None:
        values = torch.Generator().manual_seed(35)
        sample = torch.randn(7, 5, generator=values, dtype=torch.float64)
        torch.testing.assert_close(
            causal_average_matmul(sample), causal_average_loop(sample)
        )

    def test_broadcasts_the_same_causal_matrix_across_batches(self) -> None:
        generator = torch.Generator().manual_seed(350)
        values = torch.randn(3, 6, 4, generator=generator)
        actual = causal_average_matmul(values)
        expected = torch.stack([causal_average_loop(batch) for batch in values])
        torch.testing.assert_close(actual, expected)


class MaskedSoftmaxWeightTests(unittest.TestCase):
    def test_matches_normalized_triangular_weights(self) -> None:
        expected = triangular_average_weights(8, dtype=torch.float64)
        actual = masked_softmax_weights(8, dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_assigns_exact_zero_to_every_future_position(self) -> None:
        weights = masked_softmax_weights(6)
        self.assertTrue(torch.equal(weights[~causal_mask(6)], torch.zeros(15)))

    def test_rejects_nonfloating_weight_dtype(self) -> None:
        with self.assertRaisesRegex(CausalAverageError, "floating-point"):
            masked_softmax_weights(2, dtype=torch.long)


class MaskedSoftmaxAverageTests(unittest.TestCase):
    def test_matches_loop_and_triangular_matrix_methods(self) -> None:
        generator = torch.Generator().manual_seed(351)
        values = torch.randn(2, 9, 7, generator=generator, dtype=torch.float64)
        softmax_output = causal_average_softmax(values)
        torch.testing.assert_close(softmax_output, causal_average_loop(values))
        torch.testing.assert_close(softmax_output, causal_average_matmul(values))

    def test_keeps_output_on_the_input_device_and_dtype(self) -> None:
        values = torch.ones(3, 2, dtype=torch.float32)
        output = causal_average_softmax(values)
        self.assertEqual(output.device, values.device)
        self.assertEqual(output.dtype, values.dtype)


class CumulativeAverageTests(unittest.TestCase):
    def test_matches_all_curriculum_methods(self) -> None:
        generator = torch.Generator().manual_seed(352)
        values = torch.randn(4, 11, 6, generator=generator, dtype=torch.float64)
        actual = causal_average_cumsum(values)
        torch.testing.assert_close(actual, causal_average_loop(values))
        torch.testing.assert_close(actual, causal_average_matmul(values))
        torch.testing.assert_close(actual, causal_average_softmax(values))

    def test_retains_autograd_for_linear_memory_path(self) -> None:
        values = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        causal_average_cumsum(values).sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())


class EquivalenceAuditTests(unittest.TestCase):
    def test_reports_small_deterministic_forward_errors(self) -> None:
        generator = torch.Generator().manual_seed(353)
        values = torch.randn(3, 13, 5, generator=generator, dtype=torch.float64)
        first = audit_equivalence(values)
        second = audit_equivalence(values)
        self.assertEqual(first, second)
        self.assertLessEqual(first.matmul_error, 3e-16)
        self.assertLessEqual(first.softmax_error, 3e-16)
        self.assertLessEqual(first.cumsum_error, 3e-16)

    def test_gate_rejects_the_named_drifting_method(self) -> None:
        audit = CausalAverageAudit(0.0, 1e-3, 0.0)
        with self.assertRaisesRegex(CausalAverageError, "softmax forward error"):
            require_equivalence(audit, tolerance=1e-6)

    def test_gate_rejects_invalid_tolerances_and_nonfinite_errors(self) -> None:
        audit = CausalAverageAudit(0.0, 0.0, 0.0)
        with self.assertRaisesRegex(CausalAverageError, "finite and non-negative"):
            require_equivalence(audit, tolerance=float("nan"))
        with self.assertRaisesRegex(CausalAverageError, "cumsum forward error"):
            require_equivalence(
                CausalAverageAudit(0.0, 0.0, float("inf")), tolerance=1.0
            )


class WeightAuditTests(unittest.TestCase):
    def test_certifies_both_weight_constructions(self) -> None:
        for weights in (
            triangular_average_weights(7, dtype=torch.float64),
            masked_softmax_weights(7, dtype=torch.float64),
        ):
            with self.subTest():
                audit = audit_weights(weights)
                self.assertLessEqual(audit.row_sum_error, 1e-16)
                self.assertEqual(audit.maximum_future_weight, 0.0)
                self.assertEqual(audit.minimum_weight, 0.0)

    def test_exposes_future_leakage_and_negative_weights(self) -> None:
        weights = triangular_average_weights(3)
        weights[0, 2] = 0.25
        weights[1, 0] = -0.1
        audit = audit_weights(weights)
        self.assertEqual(audit.maximum_future_weight, 0.25)
        self.assertAlmostEqual(audit.minimum_weight, -0.1, places=6)

    def test_rejects_malformed_weight_matrices(self) -> None:
        with self.assertRaisesRegex(CausalAverageError, "square"):
            audit_weights(torch.ones(2, 3))
        with self.assertRaisesRegex(CausalAverageError, "finite floating-point"):
            audit_weights(torch.ones(2, 2, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
