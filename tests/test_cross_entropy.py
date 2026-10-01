"""Independent finite-difference and invariant checks for manual cross-entropy."""

import unittest

import numpy as np

from ai_journey.cross_entropy import manual_cross_entropy


class CrossEntropyTests(unittest.TestCase):
    def test_uniform_logits_have_log_class_loss(self):
        result = manual_cross_entropy(np.zeros((3, 4)), [0, 1, 3])
        self.assertAlmostEqual(result.loss, np.log(4))
        np.testing.assert_allclose(result.probabilities, 0.25)

    def test_gradient_matches_centered_finite_differences(self):
        logits = np.array([[0.2, -0.8, 1.1], [1.5, 0.4, -0.7]])
        targets = [2, 0]
        expected = manual_cross_entropy(logits, targets).dlogits
        numeric = np.zeros_like(logits)
        for index in np.ndindex(logits.shape):
            plus, minus = logits.copy(), logits.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            numeric[index] = (
                manual_cross_entropy(plus, targets).loss
                - manual_cross_entropy(minus, targets).loss
            ) / 2e-5
        np.testing.assert_allclose(expected, numeric, rtol=1e-8, atol=1e-10)

    def test_closed_form_gradient_and_row_sum(self):
        result = manual_cross_entropy([[1, 2, 3], [4, 5, 6]], [0, 2])
        expected = result.probabilities.copy()
        expected[np.arange(2), [0, 2]] -= 1
        np.testing.assert_allclose(result.dlogits, expected / 2, atol=1e-16)
        np.testing.assert_allclose(
            result.dlogits.sum(axis=1), 0, atol=np.finfo(np.float64).eps
        )
        np.testing.assert_allclose(result.dlog_normalizer, 0.5)

    def test_per_row_shift_invariance(self):
        values = np.array([[1.0, 2.0, -3.0], [-0.5, 0.2, 0.7]])
        original = manual_cross_entropy(values, [1, 0])
        shifted = manual_cross_entropy(values + [[1e4], [-1e4]], [1, 0])
        self.assertAlmostEqual(original.loss, shifted.loss, places=11)
        np.testing.assert_allclose(original.dlogits, shifted.dlogits, atol=1e-12)

    def test_underflowing_target_probability_still_has_finite_loss(self):
        result = manual_cross_entropy([[1000, -1000]], [1])
        self.assertEqual(result.loss, 2000)
        self.assertEqual(result.probabilities[0, 1], 0)
        np.testing.assert_array_equal(result.dlogits, [[1, -1]])

    def test_maximum_ties_do_not_change_derivative(self):
        result = manual_cross_entropy([[3, 3, 3]], [2])
        np.testing.assert_allclose(result.dlogits, [[1 / 3, 1 / 3, -2 / 3]])

    def test_mean_reduction_scales_duplicate_examples(self):
        single = manual_cross_entropy([[1, 2]], [0])
        repeated = manual_cross_entropy([[1, 2], [1, 2]], [0, 0])
        self.assertEqual(single.loss, repeated.loss)
        np.testing.assert_allclose(
            repeated.dlogits, np.repeat(single.dlogits / 2, 2, axis=0)
        )

    def test_inputs_are_not_mutated_or_aliased(self):
        values = np.array([[1.0, 2.0]])
        labels = np.array([0])
        result = manual_cross_entropy(values, labels)
        values[:] = 0
        labels[:] = 1
        self.assertGreater(result.loss, 1)
        self.assertFalse(np.shares_memory(result.dlogits, values))

    def test_rejects_invalid_inputs(self):
        cases = [
            ([], []),
            ([[1]], [0]),
            ([1, 2], [0]),
            ([[1, 2]], [0.5]),
            ([[1, 2]], [True]),
            ([[1, 2]], [-1]),
            ([[1, 2]], [2]),
            ([[1, 2]], [[0]]),
            ([[1, 2]], [0, 1]),
            ([[np.nan, 2]], [0]),
            ([[np.inf, 2]], [0]),
            ([[1 + 2j, 2]], [0]),
            ([["1", "2"]], [0]),
            ([[True, False]], [0]),
        ]
        for logits, labels in cases:
            with self.subTest(logits=logits, labels=labels):
                with self.assertRaises(ValueError):
                    manual_cross_entropy(logits, labels)

    def test_rejects_unrepresentable_logit_spread(self):
        with self.assertRaises(ValueError):
            manual_cross_entropy([[1e308, -1e308]], [1])


if __name__ == "__main__":
    unittest.main()
