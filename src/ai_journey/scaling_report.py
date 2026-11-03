"""Reproducible analytical reports for scaling-law preregistrations."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal

from .scaling_budget import (
    build_isoflop_sweep,
    closest_compute_optimal_candidate,
)
from .scaling_corpus import plan_corpus_reuse
from .scaling_laws import ComputeOptimalTarget
from .scaling_protocol import ScalingProtocol
from .scaling_runtime import estimate_runtime_and_cost
from .scaling_schedule import align_token_target


def _json_value(value):
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def build_scaling_report(protocol: ScalingProtocol) -> dict[str, object]:
    """Build a JSON-safe report with all derived values exposed."""

    if not isinstance(protocol, ScalingProtocol):
        raise TypeError("protocol must be ScalingProtocol")
    target = ComputeOptimalTarget(
        protocol.parameter_count,
        protocol.tokens_per_parameter,
        protocol.training_flops_per_parameter_token,
    )
    schedule = align_token_target(target.token_target, protocol.batch)
    corpus = plan_corpus_reuse(
        target_tokens=target.token_target,
        unique_train_tokens=protocol.unique_train_tokens,
    )
    candidates = build_isoflop_sweep(
        compute_budget_flops=target.training_flops,
        parameter_counts=protocol.isoflop_parameter_counts,
        optimal_tokens_per_parameter=protocol.tokens_per_parameter,
        training_flops_per_parameter_token=(
            protocol.training_flops_per_parameter_token
        ),
    )
    selected = closest_compute_optimal_candidate(candidates)
    runtime = estimate_runtime_and_cost(target.training_flops, protocol.runtime)
    return _json_value(
        {
            "schema_version": 1,
            "kind": "analytical_scaling_plan",
            "measured": False,
            "protocol": {
                "name": protocol.name,
                "parameter_count": protocol.parameter_count,
                "tokens_per_parameter": protocol.tokens_per_parameter,
                "training_flops_per_parameter_token": (
                    protocol.training_flops_per_parameter_token
                ),
                "unique_train_tokens": protocol.unique_train_tokens,
                "batch": asdict(protocol.batch),
                "maximum_overshoot_fraction": (protocol.maximum_overshoot_fraction),
                "isoflop_parameter_counts": list(protocol.isoflop_parameter_counts),
                "runtime": asdict(protocol.runtime),
            },
            "target": {
                "tokens": target.token_target,
                "training_flops": target.training_flops,
                "actual_tokens_per_parameter": (target.actual_tokens_per_parameter),
            },
            "schedule": asdict(schedule),
            "corpus": {
                **asdict(corpus),
                "repeats_data": corpus.repeats_data,
                "distinct_token_fraction": corpus.distinct_token_fraction,
            },
            "isoflop_candidates": [asdict(row) for row in candidates],
            "selected_isoflop_parameter_count": selected.parameter_count,
            "runtime_estimate": asdict(runtime),
            "evidence_boundaries": {
                "analytical_only": True,
                "training_executed": False,
                "gpu_executed": False,
                "loss_measured": False,
                "learner_completion": False,
            },
            "limitations": [
                "The token ratio is a declared planning convention, not a fitted result from this repository.",
                "The 6ND compute convention omits architecture-specific and systems overheads.",
                "Corpus epochs expose reuse but do not establish data quality or model quality.",
            ],
        }
    )
