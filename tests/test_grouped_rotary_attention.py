"""Grouped rotary math, compact cache behavior and gradients use CPU oracles."""

import copy
import math

import pytest
import torch

from ai_journey.gpt2_config import GPT2Config
from ai_journey.grouped_rotary_attention import GroupedRotarySelfAttention
from ai_journey.rope import apply_rotary_position_embeddings
from ai_journey.rotary_attention import RotarySelfAttention


def config():
    return GPT2Config(
        256,
        8,
        8,
        4,
        1,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
    )


def reference(attention, inputs):
    q, k, v = attention.c_attn(inputs).split(
        (8, attention.kv_width, attention.kv_width), -1
    )
    q = q.view(1, 4, 4, 2).transpose(1, 2)
    k = k.view(1, 4, attention.kv_heads, 2).transpose(1, 2)
    v = v.view(1, 4, attention.kv_heads, 2).transpose(1, 2)
    q, k = apply_rotary_position_embeddings(q, k)
    heads = []
    for head in range(4):
        group = head // (4 // attention.kv_heads)
        scores = q[:, head] @ k[:, group].transpose(-2, -1) / math.sqrt(2)
        scores = scores.masked_fill(
            ~torch.ones(4, 4, dtype=torch.bool).tril(), torch.finfo(scores.dtype).min
        )
        heads.append(scores.softmax(-1) @ v[:, group])
    return attention.c_proj(torch.stack(heads, 2).reshape(1, 4, 8))


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_grouped_math_and_all_gradients_match_explicit_per_query_reference(kv_heads):
    torch.manual_seed(53)
    actual = GroupedRotarySelfAttention(config(), kv_heads=kv_heads).double()
    expected = copy.deepcopy(actual)
    inputs = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    output = actual(inputs)
    oracle = reference(expected, reference_inputs)
    torch.testing.assert_close(output, oracle, rtol=1e-12, atol=1e-12)
    output.square().sum().backward()
    oracle.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    for left, right in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(left.grad, right.grad)


def test_mha_extreme_retains_existing_rotary_weights_and_numerical_behavior():
    old = RotarySelfAttention(config()).double()
    new = GroupedRotarySelfAttention(config(), kv_heads=4).double()
    new.load_state_dict(old.state_dict())
    inputs = torch.randn(1, 4, 8, dtype=torch.float64)
    torch.testing.assert_close(new(inputs), old(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_multichunk_compact_cache_matches_full_and_does_not_mutate_prefix(kv_heads):
    attention = GroupedRotarySelfAttention(config(), kv_heads=kv_heads).eval().double()
    inputs = torch.randn(2, 6, 8, dtype=torch.float64)
    full = attention(inputs)
    first, key, value = attention.forward_with_cache(inputs[:, :2])
    old_key, old_value = key.clone(), value.clone()
    middle, key2, value2 = attention.forward_with_cache(
        inputs[:, 2:5], past_key=key, past_value=value
    )
    last, _, _ = attention.forward_with_cache(
        inputs[:, 5:], past_key=key2, past_value=value2
    )
    torch.testing.assert_close(
        torch.cat((first, middle, last), 1), full, atol=1e-12, rtol=1e-12
    )
    assert key.shape == (2, kv_heads, 2, 2)
    assert torch.equal(key, old_key) and torch.equal(value, old_value)


@pytest.mark.parametrize("heads", [True, 0, 3, 5])
def test_invalid_group_contracts_are_rejected(heads):
    with pytest.raises(ValueError):
        GroupedRotarySelfAttention(config(), kv_heads=heads)


def test_score_work_budget_rejects_before_projection(monkeypatch):
    attention = GroupedRotarySelfAttention(config(), kv_heads=1)
    monkeypatch.setattr("ai_journey.grouped_rotary_attention.MAX_GQA_SCORE_ELEMENTS", 1)
    monkeypatch.setattr(
        attention.c_attn, "forward", lambda _: pytest.fail("must not project")
    )
    with pytest.raises(ValueError, match="score-work"):
        attention(torch.ones(1, 2, 8))


def test_wrong_compact_cache_and_nonfinite_projection_reject_cleanly():
    attention = GroupedRotarySelfAttention(config(), kv_heads=1).eval()
    with pytest.raises(ValueError, match="compact cache"):
        attention.forward_with_cache(
            torch.ones(1, 1, 8),
            past_key=torch.zeros(1, 4, 1, 2),
            past_value=torch.zeros(1, 4, 1, 2),
        )
    attention.c_attn.weight.data.fill_(torch.finfo(torch.float32).max)
    with pytest.raises(ValueError, match="projections"):
        attention(torch.ones(1, 1, 8))


def test_unrepresentable_rotary_base_is_a_validation_error():
    with pytest.raises(ValueError, match="rotary base"):
        GroupedRotarySelfAttention(config(), kv_heads=1, base=10**1000)
