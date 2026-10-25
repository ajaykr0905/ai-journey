from __future__ import annotations

import math

import pytest
import torch

from ai_journey.gpt2_config import GPT2Config, GPT2ConfigError
from ai_journey.rotary_attention import RotarySelfAttention


def _config() -> GPT2Config:
    return GPT2Config(
        vocab_size=17,
        context_length=5,
        embedding_dim=8,
        head_count=2,
        layer_count=1,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )


def _rotate_reference(values: torch.Tensor, base: float) -> torch.Tensor:
    rotated = torch.empty_like(values)
    head_dim = values.shape[-1]
    for position in range(values.shape[-2]):
        for feature in range(0, head_dim, 2):
            angle = position * base ** (-feature / head_dim)
            cosine, sine = math.cos(angle), math.sin(angle)
            even, odd = values[:, :, position, feature : feature + 2].unbind(-1)
            rotated[:, :, position, feature] = even * cosine - odd * sine
            rotated[:, :, position, feature + 1] = even * sine + odd * cosine
    return rotated


def test_rotary_attention_matches_independent_full_sequence_reference() -> None:
    attention = RotarySelfAttention(_config(), base=100.0).double().eval()
    with torch.no_grad():
        attention.c_attn.weight.copy_(
            torch.arange(192, dtype=torch.float64).view(24, 8) / 80
        )
        attention.c_attn.bias.copy_(torch.arange(24, dtype=torch.float64) / 100)
        attention.c_proj.weight.copy_(torch.eye(8, dtype=torch.float64))
        attention.c_proj.bias.zero_()
    inputs = torch.arange(48, dtype=torch.float64).view(2, 3, 8) / 20

    query, key, value = attention.c_attn(inputs).split(8, dim=-1)
    query = query.reshape(2, 3, 2, 4).transpose(1, 2)
    key = key.reshape(2, 3, 2, 4).transpose(1, 2)
    value = value.reshape(2, 3, 2, 4).transpose(1, 2)
    query = _rotate_reference(query, 100.0)
    key = _rotate_reference(key, 100.0)
    scores = query @ key.transpose(-2, -1) / math.sqrt(4)
    scores = scores.masked_fill(
        ~torch.tril(torch.ones(3, 3, dtype=torch.bool)),
        torch.finfo(scores.dtype).min,
    )
    expected = (torch.softmax(scores, dim=-1) @ value).transpose(1, 2)
    expected = expected.contiguous().view(2, 3, 8)

    torch.testing.assert_close(attention(inputs), expected, rtol=0, atol=1e-13)


def test_rotary_attention_does_not_let_future_tokens_affect_past_outputs() -> None:
    torch.manual_seed(52)
    attention = RotarySelfAttention(_config()).eval()
    inputs = torch.randn(1, 4, 8)
    changed = inputs.clone()
    changed[:, 3] += 100

    baseline = attention(inputs)
    perturbed = attention(changed)

    torch.testing.assert_close(baseline[:, :3], perturbed[:, :3], rtol=0, atol=0)
    assert not torch.equal(baseline[:, 3], perturbed[:, 3])


def test_rotary_attention_preserves_output_shape_and_input_gradients() -> None:
    attention = RotarySelfAttention(_config()).double()
    inputs = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)

    output = attention(inputs)
    output.square().sum().backward()

    assert output.shape == inputs.shape and output.dtype == inputs.dtype
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert attention.c_attn.weight.grad is not None
    assert torch.isfinite(attention.c_attn.weight.grad).all()


def test_rotary_attention_has_no_learned_position_table_or_persistent_mask() -> None:
    attention = RotarySelfAttention(_config())

    assert set(dict(attention.named_parameters())) == {
        "c_attn.weight",
        "c_attn.bias",
        "c_proj.weight",
        "c_proj.bias",
    }
    assert "causal_mask" not in attention.state_dict()
    assert attention.causal_mask.shape == (5, 5)
    assert attention.attn_dropout.p == _config().attention_dropout
    assert attention.resid_dropout.p == _config().residual_dropout


@pytest.mark.parametrize(
    ("inputs", "error", "message"),
    [
        ([], TypeError, "torch.Tensor"),
        (torch.ones(2, 8), GPT2ConfigError, "non-empty"),
        (torch.empty(0, 2, 8), GPT2ConfigError, "non-empty"),
        (torch.empty(1, 0, 8), GPT2ConfigError, "non-empty"),
        (torch.ones(1, 6, 8), GPT2ConfigError, "context_length"),
        (torch.ones(1, 2, 7), GPT2ConfigError, "embedding_dim"),
        (torch.ones(1, 2, 8, dtype=torch.float64), GPT2ConfigError, "parameter dtype"),
        (torch.empty(1, 2, 8, device="meta"), GPT2ConfigError, "parameter device"),
    ],
)
def test_rotary_attention_rejects_invalid_inputs(
    inputs: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        RotarySelfAttention(_config())(inputs)  # type: ignore[arg-type]


def test_rotary_attention_rejects_odd_head_dimensions() -> None:
    config = GPT2Config(
        vocab_size=17,
        context_length=5,
        embedding_dim=6,
        head_count=2,
        layer_count=1,
    )

    with pytest.raises(GPT2ConfigError, match="even"):
        RotarySelfAttention(config)


@pytest.mark.parametrize("base", [0.0, 1.0, -2.0, float("inf"), True, 10**1000])
def test_rotary_attention_rejects_invalid_bases(base: object) -> None:
    with pytest.raises(GPT2ConfigError, match="base"):
        RotarySelfAttention(_config(), base=base)  # type: ignore[arg-type]
