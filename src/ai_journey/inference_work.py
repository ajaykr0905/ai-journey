"""Analytical prefill and decode work estimates for dense decoders."""

from __future__ import annotations

from dataclasses import dataclass

from .inference_kv import _positive_integer


@dataclass(frozen=True)
class PrefillWork:
    projection_flops: int
    attention_flops: int

    @property
    def total_flops(self) -> int:
        return self.projection_flops + self.attention_flops


@dataclass(frozen=True)
class DecodeStepWork:
    projection_flops: int
    attention_flops: int

    @property
    def total_flops(self) -> int:
        return self.projection_flops + self.attention_flops


def estimate_decode_step_work(
    *,
    parameter_count: int,
    layers: int,
    query_heads: int,
    head_dim: int,
    context_tokens: int,
    batch_size: int,
) -> DecodeStepWork:
    """Estimate one cached decode step with attention linear in context."""

    values = {
        "parameter_count": parameter_count,
        "layers": layers,
        "query_heads": query_heads,
        "head_dim": head_dim,
        "context_tokens": context_tokens,
        "batch_size": batch_size,
    }
    for name, value in values.items():
        _positive_integer(value, name)
    projection = 2 * parameter_count * batch_size
    attention = 4 * layers * batch_size * query_heads * head_dim * context_tokens
    return DecodeStepWork(projection, attention)


def estimate_prefill_work(
    *,
    parameter_count: int,
    layers: int,
    query_heads: int,
    head_dim: int,
    sequence_tokens: int,
    batch_size: int,
) -> PrefillWork:
    """Estimate forward FLOPs with an explicit quadratic attention term."""

    values = {
        "parameter_count": parameter_count,
        "layers": layers,
        "query_heads": query_heads,
        "head_dim": head_dim,
        "sequence_tokens": sequence_tokens,
        "batch_size": batch_size,
    }
    for name, value in values.items():
        _positive_integer(value, name)
    tokens = sequence_tokens * batch_size
    projection = 2 * parameter_count * tokens
    attention = 4 * layers * batch_size * query_heads * head_dim * sequence_tokens**2
    return PrefillWork(projection, attention)
