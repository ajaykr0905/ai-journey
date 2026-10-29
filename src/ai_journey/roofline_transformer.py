"""Roofline workloads for one dense decoder microbatch."""

from __future__ import annotations

from .resource_accounting import DecoderSpec, ResourceAccountingError
from .roofline import (
    KernelWorkload,
    RooflineEstimate,
    RooflineHardware,
    analyze_roofline,
    layer_norm_workload,
    matmul_workload,
)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ResourceAccountingError(f"{name} must be positive")
    return value


def decoder_microbatch_workloads(
    model: DecoderSpec,
    *,
    batch_size: int,
    sequence_length: int,
    dtype: str,
) -> dict[str, KernelWorkload]:
    """Build named kernel-family estimates for a dense decoder forward pass.

    Identically shaped layer/head operations are aggregated through
    ``batch_count``. This preserves arithmetic intensity while retaining the
    complete modeled forward matmul work.
    """

    if not isinstance(model, DecoderSpec):
        raise TypeError("model must be a DecoderSpec")
    batch = _positive_integer(batch_size, "batch_size")
    sequence = _positive_integer(sequence_length, "sequence_length")
    tokens = batch * sequence
    layers = model.layer_count
    hidden = model.hidden_size
    return {
        "attention_query": matmul_workload(
            rows=tokens,
            inner=hidden,
            columns=hidden,
            batch_count=layers,
            dtype=dtype,
        ),
        "attention_key_value": matmul_workload(
            rows=tokens,
            inner=hidden,
            columns=2 * model.kv_size,
            batch_count=layers,
            dtype=dtype,
        ),
        "attention_output": matmul_workload(
            rows=tokens,
            inner=hidden,
            columns=hidden,
            batch_count=layers,
            dtype=dtype,
        ),
        "attention_scores": matmul_workload(
            rows=sequence,
            inner=model.head_size,
            columns=sequence,
            batch_count=batch * layers * model.attention_heads,
            dtype=dtype,
        ),
        "attention_value_mix": matmul_workload(
            rows=sequence,
            inner=sequence,
            columns=model.head_size,
            batch_count=batch * layers * model.attention_heads,
            dtype=dtype,
        ),
        "feed_forward_gate_up": matmul_workload(
            rows=tokens,
            inner=hidden,
            columns=2 * model.feed_forward_size,
            batch_count=layers,
            dtype=dtype,
        ),
        "feed_forward_down": matmul_workload(
            rows=tokens,
            inner=model.feed_forward_size,
            columns=hidden,
            batch_count=layers,
            dtype=dtype,
        ),
        "block_layer_norms": layer_norm_workload(
            tokens=tokens * layers * 2,
            hidden_size=hidden,
            dtype=dtype,
        ),
        "final_layer_norm": layer_norm_workload(
            tokens=tokens,
            hidden_size=hidden,
            dtype=dtype,
        ),
        "output_projection": matmul_workload(
            rows=tokens,
            inner=hidden,
            columns=model.vocab_size,
            dtype=dtype,
        ),
    }


def analyze_decoder_microbatch(
    model: DecoderSpec,
    *,
    batch_size: int,
    sequence_length: int,
    dtype: str,
    hardware: RooflineHardware,
) -> dict[str, RooflineEstimate]:
    """Classify every modeled decoder kernel family against one roofline."""

    if not isinstance(hardware, RooflineHardware):
        raise TypeError("hardware must be RooflineHardware")
    return {
        name: analyze_roofline(workload, hardware)
        for name, workload in decoder_microbatch_workloads(
            model,
            batch_size=batch_size,
            sequence_length=sequence_length,
            dtype=dtype,
        ).items()
    }
