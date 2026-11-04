"""Analytical prefill and decode work estimates for dense decoders."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .inference_kv import KVCacheLayout, _positive_integer


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


@dataclass(frozen=True)
class DecodeMemoryTraffic:
    weight_bytes: int
    kv_read_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.kv_read_bytes

    def arithmetic_intensity(self, work: DecodeStepWork) -> Fraction:
        if not isinstance(work, DecodeStepWork):
            raise TypeError("work must be DecodeStepWork")
        return Fraction(work.total_flops, self.total_bytes)


def estimate_decode_memory_traffic(
    *, parameter_count: int, weight_bytes_per_parameter: int, cache: KVCacheLayout
) -> DecodeMemoryTraffic:
    """Estimate compulsory weight and resident KV reads for one decode step."""

    _positive_integer(parameter_count, "parameter_count")
    _positive_integer(weight_bytes_per_parameter, "weight_bytes_per_parameter")
    if not isinstance(cache, KVCacheLayout):
        raise TypeError("cache must be KVCacheLayout")
    return DecodeMemoryTraffic(
        weight_bytes=parameter_count * weight_bytes_per_parameter,
        kv_read_bytes=cache.total_bytes,
    )


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
