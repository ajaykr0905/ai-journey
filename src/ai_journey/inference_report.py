"""Deterministic analytical reports for decoder inference-capacity plans."""

from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction

from .inference_capacity import admit_sequence_lengths
from .inference_kv import KVCacheLayout, grouped_query_cache_fraction
from .inference_paging import plan_paged_allocation
from .inference_protocol import InferenceProtocol
from .inference_work import estimate_decode_step_work, estimate_prefill_work


def _json_value(value):
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def build_inference_report(protocol: InferenceProtocol) -> dict[str, object]:
    """Build a JSON-safe report without claiming execution or measurements."""

    if not isinstance(protocol, InferenceProtocol):
        raise TypeError("protocol must be InferenceProtocol")
    layout = protocol.kv_layout
    mha = KVCacheLayout(
        protocol.layers,
        protocol.query_heads,
        protocol.head_dim,
        protocol.context_tokens,
        len(protocol.request_lengths),
        protocol.bytes_per_element,
    )
    admission = admit_sequence_lengths(
        protocol.device_budget, layout, protocol.request_lengths
    )
    paging = plan_paged_allocation(
        protocol.request_lengths,
        block_tokens=protocol.block_tokens,
        bytes_per_token=layout.bytes_per_token_per_sequence,
    )
    prefill = estimate_prefill_work(
        parameter_count=protocol.parameter_count,
        layers=protocol.layers,
        query_heads=protocol.query_heads,
        head_dim=protocol.head_dim,
        sequence_tokens=protocol.context_tokens,
        batch_size=len(protocol.request_lengths),
    )
    decode = estimate_decode_step_work(
        parameter_count=protocol.parameter_count,
        layers=protocol.layers,
        query_heads=protocol.query_heads,
        head_dim=protocol.head_dim,
        context_tokens=protocol.context_tokens,
        batch_size=len(protocol.request_lengths),
    )
    return _json_value(
        {
            "schema_version": 1,
            "kind": "analytical_inference_capacity_plan",
            "measured": False,
            "protocol": asdict(protocol),
            "kv_cache": {
                "bytes_per_token_per_sequence": layout.bytes_per_token_per_sequence,
                "gqa_total_bytes": layout.total_bytes,
                "mha_total_bytes": mha.total_bytes,
                "gqa_fraction_of_mha": grouped_query_cache_fraction(
                    query_heads=protocol.query_heads, kv_heads=protocol.kv_heads
                ),
            },
            "device_budget": asdict(protocol.device_budget),
            "ragged_admission": asdict(admission),
            "paged_allocation": {
                **asdict(paging),
                "internal_waste_fraction": paging.internal_waste_fraction,
            },
            "prefill_work": {**asdict(prefill), "total_flops": prefill.total_flops},
            "decode_step_work": {
                **asdict(decode),
                "total_flops": decode.total_flops,
            },
            "speculative": {
                **asdict(protocol.speculative),
                "acceptance_probability": protocol.speculative.acceptance_probability,
                "expected_accepted_draft_tokens": (
                    protocol.speculative.expected_accepted_draft_tokens
                ),
                "expected_output_tokens_per_target_pass": (
                    protocol.speculative.expected_output_tokens_per_target_pass
                ),
            },
            "evidence_boundaries": {
                "analytical_only": True,
                "gpu_executed": False,
                "server_executed": False,
                "latency_measured": False,
                "learner_completion": False,
            },
            "limitations": [
                "FLOP formulas are analytical conventions and omit kernel and scheduler overhead.",
                "Byte accounting does not establish allocator residency or usable device capacity.",
                "Speculative acceptance is declared, not measured on a workload.",
            ],
        }
    )
