"""Deterministic reports for analytical decoder roofline plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from .resource_accounting import DecoderSpec, ResourceAccountingError
from .roofline import RooflineHardware
from .roofline_transformer import analyze_decoder_microbatch


@dataclass(frozen=True)
class RooflinePlan:
    """Inputs required to reproduce one decoder roofline estimate."""

    model: DecoderSpec
    batch_size: int
    sequence_length: int
    dtype: str
    peak_flops_per_second: float
    memory_bandwidth_bytes_per_second: float


def build_roofline_report(plan: RooflinePlan) -> dict[str, object]:
    """Build a JSON-safe report from validated, replayable assumptions."""

    if not isinstance(plan, RooflinePlan):
        raise TypeError("plan must be a RooflinePlan")
    hardware = RooflineHardware(
        peak_flops_per_second=plan.peak_flops_per_second,
        memory_bandwidth_bytes_per_second=plan.memory_bandwidth_bytes_per_second,
    )
    estimates = analyze_decoder_microbatch(
        plan.model,
        batch_size=plan.batch_size,
        sequence_length=plan.sequence_length,
        dtype=plan.dtype,
        hardware=hardware,
    )
    operations = {}
    counts = {"compute": 0, "memory": 0, "ridge": 0}
    for name, estimate in estimates.items():
        counts[estimate.bottleneck] += 1
        workload = estimate.workload
        operations[name] = {
            "kind": workload.kind,
            "flops": workload.flops,
            "bytes_read": workload.bytes_read,
            "bytes_written": workload.bytes_written,
            "total_bytes": workload.total_bytes,
            "arithmetic_intensity": workload.arithmetic_intensity,
            "compute_seconds": estimate.compute_seconds,
            "memory_seconds": estimate.memory_seconds,
            "lower_bound_seconds": estimate.lower_bound_seconds,
            "attainable_flops_per_second": estimate.attainable_flops_per_second,
            "bottleneck": estimate.bottleneck,
        }
    return {
        "schema_version": 1,
        "kind": "analytical_decoder_roofline",
        "measured": False,
        "model": asdict(plan.model),
        "workload": {
            "batch_size": plan.batch_size,
            "sequence_length": plan.sequence_length,
            "dtype": plan.dtype,
        },
        "hardware": {
            **asdict(hardware),
            "ridge_point_flops_per_byte": hardware.ridge_point,
        },
        "operations": operations,
        "summary": {
            "operation_count": len(operations),
            "bottleneck_counts": counts,
            "total_flops": sum(item["flops"] for item in operations.values()),
            "total_compulsory_bytes": sum(
                item["total_bytes"] for item in operations.values()
            ),
            "serial_lower_bound_seconds": math.fsum(
                item["lower_bound_seconds"] for item in operations.values()
            ),
        },
        "limitations": [
            "Analytical estimate only; no accelerator or kernel was measured.",
            "Compulsory tensor traffic excludes cache effects, fusion, allocator traffic, and spills.",
            "Serial lower-bound time sums isolated operation ceilings and excludes overlap and launch overhead.",
        ],
    }


def _canonical_report_bytes(report: dict[str, object]) -> bytes:
    unsigned = {key: value for key, value in report.items() if key != "integrity"}
    return json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_roofline_report(report: dict[str, object]) -> dict[str, object]:
    """Attach a deterministic SHA-256 digest without mutating the input."""

    sealed = {key: value for key, value in report.items() if key != "integrity"}
    sealed["integrity"] = {
        "algorithm": "sha256",
        "digest": hashlib.sha256(_canonical_report_bytes(sealed)).hexdigest(),
    }
    return sealed


def verify_roofline_report(report: dict[str, object]) -> dict[str, object]:
    """Recompute every derived field and reject report drift."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ResourceAccountingError("unsupported roofline report schema")
    if report.get("kind") != "analytical_decoder_roofline":
        raise ResourceAccountingError("unexpected roofline report kind")
    if report.get("measured") is not False:
        raise ResourceAccountingError("roofline report must remain unmeasured")
    try:
        actual_integrity = seal_roofline_report(report)["integrity"]
    except (TypeError, ValueError, OverflowError) as error:
        raise ResourceAccountingError("roofline report cannot be recomputed") from error
    if report.get("integrity") != actual_integrity:
        raise ResourceAccountingError(
            "roofline report integrity does not match recomputation"
        )
    try:
        model_payload = report["model"]
        workload = report["workload"]
        hardware = report["hardware"]
        if not all(
            isinstance(value, dict) for value in (model_payload, workload, hardware)
        ):
            raise TypeError("roofline plan sections must be dictionaries")
        plan = RooflinePlan(
            model=DecoderSpec(**model_payload),
            batch_size=workload["batch_size"],
            sequence_length=workload["sequence_length"],
            dtype=workload["dtype"],
            peak_flops_per_second=hardware["peak_flops_per_second"],
            memory_bandwidth_bytes_per_second=hardware[
                "memory_bandwidth_bytes_per_second"
            ],
        )
    except (KeyError, TypeError, ResourceAccountingError) as error:
        raise ResourceAccountingError("invalid roofline report plan") from error
    expected = seal_roofline_report(build_roofline_report(plan))
    if report != expected:
        raise ResourceAccountingError("roofline report does not match recomputation")
    return report
