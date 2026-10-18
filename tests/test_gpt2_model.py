from __future__ import annotations

import math

import pytest
import torch

from ai_journey.gpt2_config import GPT2Config, GPT2ConfigError
from ai_journey.gpt2_model import GPT2MLP, GPT2SelfAttention, gelu_new


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


def _tiny_config() -> GPT2Config:
    return GPT2Config(
        vocab_size=11,
        context_length=4,
        embedding_dim=4,
        head_count=2,
        layer_count=1,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )


def test_attention_matches_an_explicit_causal_reference() -> None:
    attention = GPT2SelfAttention(_tiny_config()).double().eval()
    with torch.no_grad():
        attention.c_attn.weight.copy_(
            torch.arange(48, dtype=torch.float64).view(12, 4) / 50
        )
        attention.c_attn.bias.copy_(torch.arange(12, dtype=torch.float64) / 100)
        attention.c_proj.weight.copy_(torch.eye(4, dtype=torch.float64))
        attention.c_proj.bias.zero_()
    inputs = torch.arange(24, dtype=torch.float64).view(2, 3, 4) / 10

    query, key, value = attention.c_attn(inputs).split(4, dim=2)
    query = query.view(2, 3, 2, 2).transpose(1, 2)
    key = key.view(2, 3, 2, 2).transpose(1, 2)
    value = value.view(2, 3, 2, 2).transpose(1, 2)
    scores = query @ key.transpose(-2, -1) / math.sqrt(2)
    scores = scores.masked_fill(
        ~torch.tril(torch.ones(3, 3, dtype=torch.bool)),
        torch.finfo(torch.float64).min,
    )
    expected = (torch.softmax(scores, dim=-1) @ value).transpose(1, 2)
    expected = expected.contiguous().view(2, 3, 4)

    torch.testing.assert_close(attention(inputs), expected, rtol=0, atol=1e-14)


def test_attention_is_causal() -> None:
    torch.manual_seed(45)
    attention = GPT2SelfAttention(_tiny_config()).eval()
    inputs = torch.randn(1, 4, 4)
    changed = inputs.clone()
    changed[:, 3] += 100

    baseline = attention(inputs)
    perturbed = attention(changed)

    torch.testing.assert_close(baseline[:, :3], perturbed[:, :3])
    assert not torch.equal(baseline[:, 3], perturbed[:, 3])


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (torch.ones(4), "non-empty"),
        (torch.ones(1, 5, 4), "context_length"),
        (torch.ones(1, 2, 3), "embedding_dim"),
        (torch.ones(1, 2, 4, dtype=torch.float64), "parameter dtype"),
    ],
)
def test_attention_rejects_invalid_input_contracts(
    inputs: torch.Tensor, message: str
) -> None:
    with pytest.raises(GPT2ConfigError, match=message):
        GPT2SelfAttention(_tiny_config())(inputs)


def test_mlp_matches_explicit_expand_activate_project_equation() -> None:
    mlp = GPT2MLP(_tiny_config()).double().eval()
    with torch.no_grad():
        mlp.c_fc.weight.copy_(torch.arange(64, dtype=torch.float64).view(16, 4) / 100)
        mlp.c_fc.bias.copy_(torch.arange(16, dtype=torch.float64) / 100)
        mlp.c_proj.weight.copy_(torch.arange(64, dtype=torch.float64).view(4, 16) / 200)
        mlp.c_proj.bias.copy_(torch.arange(4, dtype=torch.float64) / 100)
    inputs = torch.arange(24, dtype=torch.float64).view(2, 3, 4) / 20

    expected = torch.nn.functional.linear(
        gelu_new(torch.nn.functional.linear(inputs, mlp.c_fc.weight, mlp.c_fc.bias)),
        mlp.c_proj.weight,
        mlp.c_proj.bias,
    )

    torch.testing.assert_close(mlp(inputs), expected, rtol=0, atol=0)


def test_mlp_uses_a_four_times_expansion() -> None:
    mlp = GPT2MLP(_tiny_config())

    assert mlp.c_fc.weight.shape == (16, 4)
    assert mlp.c_proj.weight.shape == (4, 16)


def test_mlp_rejects_wrong_rank_width_and_dtype() -> None:
    mlp = GPT2MLP(_tiny_config())

    with pytest.raises(GPT2ConfigError, match="shape"):
        mlp(torch.ones(2, 4))
    with pytest.raises(GPT2ConfigError, match="shape"):
        mlp(torch.ones(1, 2, 3))
    with pytest.raises(GPT2ConfigError, match="dtype"):
        mlp(torch.ones(1, 2, 4, dtype=torch.float64))
