from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from ai_journey.modern_architecture import RMSNorm


def test_rms_norm_matches_pytorch_reference_forward_and_gradients() -> None:
    inputs = torch.tensor(
        [[[3.0, 4.0, -2.0, 1.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_inputs = inputs.detach().clone().requires_grad_()
    norm = RMSNorm(4, eps=1e-6).to(dtype=torch.float64)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 0.5, -2.0, 1.5]))
    reference_weight = norm.weight.detach().clone().requires_grad_()

    actual = norm(inputs)
    expected = F.rms_norm(reference_inputs, (4,), reference_weight, eps=1e-6)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[0, 1], torch.zeros(4, dtype=torch.float64))

    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    torch.testing.assert_close(norm.weight.grad, reference_weight.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_preserves_input_shape_and_dtype(dtype: torch.dtype) -> None:
    norm = RMSNorm(4).to(dtype=dtype)
    inputs = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=dtype)

    result = norm(inputs)

    assert result.shape == inputs.shape
    assert result.dtype == dtype
    assert torch.isfinite(result).all()


def test_rms_norm_avoids_half_precision_square_overflow() -> None:
    norm = RMSNorm(2).to(dtype=torch.float16)
    inputs = torch.tensor([[300.0, 400.0]], dtype=torch.float16)

    result = norm(inputs)

    torch.testing.assert_close(
        result,
        torch.tensor([[0.8485, 1.1314]], dtype=torch.float16),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_avoids_float32_accumulator_square_overflow(
    dtype: torch.dtype,
) -> None:
    norm = RMSNorm(2).to(dtype=dtype)
    inputs = torch.tensor([[1e20, -2e20]], dtype=dtype)

    result = norm(inputs)
    expected = F.rms_norm(inputs.double(), (2,), norm.weight.double(), eps=norm.eps).to(
        dtype=dtype
    )

    assert result.dtype == dtype
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("width", [0, -1, True, 2.5])
def test_rms_norm_rejects_invalid_width(width: object) -> None:
    with pytest.raises((TypeError, ValueError), match="width"):
        RMSNorm(width)  # type: ignore[arg-type]


@pytest.mark.parametrize("eps", [0.0, -1e-6, float("inf"), float("nan")])
def test_rms_norm_rejects_invalid_epsilon(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        RMSNorm(4, eps=eps)


def test_rms_norm_rejects_wrong_feature_width_and_nonfloating_input() -> None:
    norm = RMSNorm(4)

    with pytest.raises(ValueError, match="last dimension"):
        norm(torch.ones(2, 3))
    with pytest.raises(TypeError, match="floating"):
        norm(torch.ones(2, 4, dtype=torch.int64))
