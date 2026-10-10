from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from ai_journey.transformer_lab import (
    CausalSelfAttention,
    TransformerConfig,
    TransformerLabError,
)


class AttentionTensorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.attention = CausalSelfAttention(
            TransformerConfig(vocab_size=7, block_size=5, embedding_dim=8, head_count=2)
        )

    def test_valid_float64_and_noncontiguous_inputs_keep_gradients(self) -> None:
        self.attention.double()
        inputs = torch.randn(2, 8, 4, dtype=torch.float64).transpose(1, 2)
        inputs.requires_grad_()
        output = self.attention(inputs)
        self.assertEqual(output.shape, (2, 4, 8))
        self.assertEqual(output.dtype, torch.float64)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_rejects_invalid_shapes_before_projection(self) -> None:
        for inputs in (
            torch.ones(4, 8),
            torch.ones(0, 4, 8),
            torch.ones(2, 0, 8),
            torch.ones(2, 4, 7),
            torch.ones(2, 6, 8),
        ):
            with self.subTest(shape=inputs.shape), self.assertRaises(
                TransformerLabError
            ):
                self.attention(inputs)

    def test_rejects_nonfloating_and_mismatched_dtypes(self) -> None:
        with self.assertRaisesRegex(TypeError, "floating-point"):
            self.attention(torch.ones(2, 4, 8, dtype=torch.long))
        with self.assertRaisesRegex(TransformerLabError, "dtype and device"):
            self.attention(torch.ones(2, 4, 8, dtype=torch.float64))
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            self.attention([[[1.0] * 8]])

    def test_rejects_nonfinite_inputs_and_projection_overflow(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf")):
            inputs = torch.ones(2, 4, 8)
            inputs[0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                TransformerLabError, "finite"
            ):
                self.attention(inputs)
        with torch.no_grad():
            self.attention.query_key_value.weight.fill_(torch.finfo(torch.float32).max)
        with self.assertRaisesRegex(TransformerLabError, "projections must be finite"):
            self.attention(torch.ones(2, 4, 8))

    def test_rejects_device_mismatch_without_allocating_device_storage(self) -> None:
        self.attention.to("meta")
        with self.assertRaisesRegex(TransformerLabError, "dtype and device"):
            self.attention(torch.ones(2, 4, 8))
