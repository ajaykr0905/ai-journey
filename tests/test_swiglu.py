from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from ai_journey.swiglu import SwiGLU


def test_swiglu_matches_independent_projection_reference() -> None:
    layer = SwiGLU(4, 7, bias=True).double()
    with torch.no_grad():
        for index, projection in enumerate(
            (layer.gate_proj, layer.up_proj, layer.down_proj), start=1
        ):
            projection.weight.copy_(
                torch.arange(projection.weight.numel(), dtype=torch.float64)
                .reshape_as(projection.weight)
                .div_(100 * index)
            )
            assert projection.bias is not None
            projection.bias.copy_(
                torch.arange(projection.out_features, dtype=torch.float64).div_(50)
            )
    inputs = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4).div_(20)

    gate = F.silu(F.linear(inputs, layer.gate_proj.weight, layer.gate_proj.bias))
    values = F.linear(inputs, layer.up_proj.weight, layer.up_proj.bias)
    expected = F.linear(gate * values, layer.down_proj.weight, layer.down_proj.bias)

    torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4)])
def test_swiglu_preserves_leading_shape_and_parameter_dtype(
    shape: tuple[int, ...],
) -> None:
    layer = SwiGLU(4, 6).double()
    inputs = torch.randn(shape, dtype=torch.float64)

    output = layer(inputs)

    assert output.shape == inputs.shape
    assert output.dtype == inputs.dtype


def test_swiglu_has_three_bias_free_projections_with_explicit_hidden_width() -> None:
    layer = SwiGLU(4, 6)

    assert layer.embedding_dim == 4
    assert layer.hidden_dim == 6
    assert layer.gate_proj.weight.shape == (6, 4)
    assert layer.up_proj.weight.shape == (6, 4)
    assert layer.down_proj.weight.shape == (4, 6)
    assert layer.gate_proj.bias is None
    assert layer.up_proj.bias is None
    assert layer.down_proj.bias is None
    assert sum(parameter.numel() for parameter in layer.parameters()) == 72


def test_swiglu_backpropagates_through_input_and_all_projections() -> None:
    torch.manual_seed(52)
    layer = SwiGLU(3, 5).double()
    inputs = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

    layer(inputs).square().sum().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert torch.count_nonzero(inputs.grad) > 0
    for projection in (layer.gate_proj, layer.up_proj, layer.down_proj):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert torch.count_nonzero(projection.weight.grad) > 0


@pytest.mark.parametrize(
    ("embedding_dim", "hidden_dim"),
    [(0, 4), (-1, 4), (4, 0), (4, -1), (True, 4), (4, False), (4.0, 4)],
)
def test_swiglu_rejects_nonpositive_and_noninteger_dimensions(
    embedding_dim: int, hidden_dim: int
) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        SwiGLU(embedding_dim, hidden_dim)


def test_swiglu_rejects_nonboolean_bias() -> None:
    with pytest.raises(TypeError, match="bias"):
        SwiGLU(4, 6, bias=1)  # type: ignore[arg-type]


def test_swiglu_rejects_nonfloating_or_wrong_width_inputs() -> None:
    layer = SwiGLU(4, 6)

    with pytest.raises(TypeError, match="torch.Tensor"):
        layer([1.0, 2.0, 3.0, 4.0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        layer(torch.ones(2, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="embedding_dim"):
        layer(torch.ones(2, 3))
    with pytest.raises(ValueError, match="embedding_dim"):
        layer(torch.tensor(1.0))
