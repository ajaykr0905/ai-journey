from __future__ import annotations

import math

import pytest
import torch

from ai_journey.gpt2_model import gelu_new


def test_gelu_new_matches_the_published_tanh_equation() -> None:
    inputs = torch.tensor([-3.0, -1.0, 0.0, 0.5, 2.0], dtype=torch.float64)
    expected = (
        0.5
        * inputs
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (inputs + 0.044715 * inputs.pow(3))))
    )

    torch.testing.assert_close(gelu_new(inputs), expected, rtol=0, atol=0)


def test_gelu_new_preserves_shape_dtype_and_gradients() -> None:
    inputs = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)

    output = gelu_new(inputs)
    output.sum().backward()

    assert output.shape == inputs.shape
    assert output.dtype == inputs.dtype
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_gelu_new_rejects_non_tensor_and_integer_inputs() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        gelu_new([1.0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        gelu_new(torch.tensor([1]))
