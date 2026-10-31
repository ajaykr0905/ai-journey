"""Deterministic evidence reports for analytical ZeRO stage comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from .distributed_capacity import ClusterTopology, LinkProfile
from .resource_accounting import DecoderSpec
from .zero_planning import (
    admit_zero_stage,
    estimate_zero_collectives,
    zero_stage_memory_peaks,
    zero_stage_state,
)


class ZeroReportError(ValueError):
    """Raised when ZeRO comparison evidence cannot be rebuilt."""


@dataclass(frozen=True)
class ZeroComparisonPlan:
    """Declared inputs shared by every ZeRO stage in one comparison."""

    model: DecoderSpec
    topology: ClusterTopology
    parameter_dtype: str
    gradient_dtype: str
    optimizer_dtype: str
    optimizer_slots: int
    master_parameter_dtype: str | None
    activation_bytes: int
    gradient_bucket_bytes: int
    device_capacity_bytes: int
    reserve_fraction: float


def build_zero_comparison_report(plan: ZeroComparisonPlan) -> dict[str, object]:
    """Build one JSON-safe, unmeasured comparison for ZeRO stages 0 through 3."""

    if not isinstance(plan, ZeroComparisonPlan):
        raise TypeError("plan must be a ZeroComparisonPlan")
    stages = []
    for stage in range(4):
        states = zero_stage_state(
            plan.model,
            stage=stage,
            world_size=plan.topology.world_size,
            parameter_dtype=plan.parameter_dtype,
            gradient_dtype=plan.gradient_dtype,
            optimizer_dtype=plan.optimizer_dtype,
            optimizer_slots=plan.optimizer_slots,
            master_parameter_dtype=plan.master_parameter_dtype,
        )
        memory = zero_stage_memory_peaks(
            plan.model,
            stage=stage,
            world_size=plan.topology.world_size,
            parameter_dtype=plan.parameter_dtype,
            gradient_dtype=plan.gradient_dtype,
            optimizer_dtype=plan.optimizer_dtype,
            optimizer_slots=plan.optimizer_slots,
            master_parameter_dtype=plan.master_parameter_dtype,
            activation_bytes=plan.activation_bytes,
            gradient_bucket_bytes=plan.gradient_bucket_bytes,
        )
        collectives = estimate_zero_collectives(
            plan.model,
            stage=stage,
            topology=plan.topology,
            parameter_dtype=plan.parameter_dtype,
            gradient_dtype=plan.gradient_dtype,
        )
        admission = admit_zero_stage(
            memory,
            device_capacity_bytes=plan.device_capacity_bytes,
            reserve_fraction=plan.reserve_fraction,
        )
        stages.append(
            {
                "stage": stage,
                "rank_states": [
                    {**asdict(state), "total_bytes": state.total_bytes}
                    for state in states
                ],
                "rank_memory": [asdict(item) for item in memory],
                "collectives": [asdict(item) for item in collectives],
                "admission": asdict(admission),
                "max_peak_bytes": max(item.peak_bytes for item in memory),
                "wire_bytes_per_rank": sum(
                    item.wire_bytes_per_rank for item in collectives
                ),
                "seconds": math.fsum(item.seconds for item in collectives),
            }
        )
    report = {
        "schema_version": 1,
        "kind": "analytical_zero_stage_comparison",
        "measured": False,
        "model": asdict(plan.model),
        "topology": asdict(plan.topology),
        "assumptions": {
            key: value
            for key, value in asdict(plan).items()
            if key not in {"model", "topology"}
        },
        "stages": stages,
        "limitations": [
            "Analytical comparison only; no GPU allocation or collective was measured.",
            "Ring time uses declared bandwidth and latency assumptions.",
            "Parameter and gradient transient peaks are modeled as non-overlapping.",
            "Allocator fragmentation and runtime workspaces are excluded.",
        ],
    }
    return json.loads(json.dumps(report, allow_nan=False))


def _canonical_report_bytes(report: dict[str, object]) -> bytes:
    unsigned = {key: value for key, value in report.items() if key != "integrity"}
    return json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_zero_comparison_report(report: dict[str, object]) -> dict[str, object]:
    """Attach a deterministic SHA-256 seal without mutating the input."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    sealed = {key: value for key, value in report.items() if key != "integrity"}
    sealed["integrity"] = {
        "algorithm": "sha256",
        "digest": hashlib.sha256(_canonical_report_bytes(sealed)).hexdigest(),
    }
    return sealed


def verify_zero_comparison_report(report: dict[str, object]) -> dict[str, object]:
    """Verify the seal and rebuild every stage from declared inputs."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    if report.get("schema_version") != 1:
        raise ZeroReportError("unsupported report schema")
    if report.get("kind") != "analytical_zero_stage_comparison":
        raise ZeroReportError("unexpected report kind")
    if report.get("measured") is not False:
        raise ZeroReportError("ZeRO comparison report must remain unmeasured")
    try:
        expected_integrity = seal_zero_comparison_report(report)["integrity"]
    except (TypeError, ValueError, OverflowError) as error:
        raise ZeroReportError("ZeRO comparison report is not canonical") from error
    if report.get("integrity") != expected_integrity:
        raise ZeroReportError("ZeRO comparison report integrity mismatch")
    try:
        model_payload = report["model"]
        topology_payload = report["topology"]
        assumptions = report["assumptions"]
        if not all(
            isinstance(item, dict)
            for item in (model_payload, topology_payload, assumptions)
        ):
            raise TypeError("plan sections must be objects")
        topology_payload = dict(topology_payload)
        topology = ClusterTopology(
            world_size=topology_payload["world_size"],
            workers_per_node=topology_payload["workers_per_node"],
            intra_node=LinkProfile(**topology_payload["intra_node"]),
            inter_node=LinkProfile(**topology_payload["inter_node"]),
        )
        plan = ZeroComparisonPlan(
            model=DecoderSpec(**model_payload),
            topology=topology,
            **assumptions,
        )
        expected = seal_zero_comparison_report(build_zero_comparison_report(plan))
    except (KeyError, TypeError, ValueError) as error:
        raise ZeroReportError("ZeRO comparison report cannot be rebuilt") from error
    if report != expected:
        raise ZeroReportError("ZeRO comparison report derived fields drifted")
    return report
