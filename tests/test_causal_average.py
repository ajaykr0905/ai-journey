from __future__ import annotations

import unittest

import torch

from ai_journey.causal_average import CausalAverageError, causal_mask, validate_values


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


if __name__ == "__main__":
    unittest.main()
