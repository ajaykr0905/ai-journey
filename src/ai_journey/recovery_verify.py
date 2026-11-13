"""Independent verification for sealed recovery experiment reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from .recovery_config import RecoveryExperimentError
from .recovery_metrics import measure_recovery_timing
from .recovery_trace_io import load_recovery_trace

_FIELDS = {
    "schema_version",
    "experiment",
    "attempts",
    "fault",
    "loss_assessment",
    "timing",
    "trace_sha256",
    "evidence_boundary",
    "report_sha256",
}


def verify_recovery_report(
    report: Mapping[str, Any], *, trace_path: Path | None = None
) -> dict[str, Any]:
    """Recompute the report seal, accounting identities, timings, and trace binding."""

    if not isinstance(report, Mapping) or set(report) != _FIELDS:
        raise RecoveryExperimentError("recovery report fields do not match schema")
    if report["schema_version"] != 1:
        raise RecoveryExperimentError("recovery report schema version is unsupported")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    expected_digest = sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if report["report_sha256"] != expected_digest:
        raise RecoveryExperimentError("recovery report digest does not match")
    experiment = report["experiment"]
    attempts = report["attempts"]
    fault = report["fault"]
    loss = report["loss_assessment"]
    timing = report["timing"]
    if not all(
        isinstance(item, Mapping)
        for item in (experiment, attempts, fault, loss, timing)
    ):
        raise RecoveryExperimentError("recovery report sections must be mappings")
    if set(experiment) != {
        "run_id",
        "config_sha256",
        "workload_sha256",
        "source_bytes",
        "source_sha256",
        "work_items",
    }:
        raise RecoveryExperimentError("experiment fields do not match schema")
    if set(attempts) != {
        "failed_attempt_id",
        "resumed_attempt_id",
        "failed_exit_code",
        "resumed_exit_code",
        "parent_checkpoint_sha256",
    }:
        raise RecoveryExperimentError("attempt fields do not match schema")
    if attempts["failed_attempt_id"] == attempts["resumed_attempt_id"]:
        raise RecoveryExperimentError(
            "report does not contain a distinct resumed attempt"
        )
    if attempts["failed_exit_code"] == 0 or attempts["resumed_exit_code"] != 0:
        raise RecoveryExperimentError(
            "report exit codes do not prove failure and recovery"
        )
    if set(fault) != {"mechanism", "fault_after_items", "checkpoint_interval"}:
        raise RecoveryExperimentError("fault fields do not match schema")
    if fault["mechanism"] != "process_kill":
        raise RecoveryExperimentError("fault mechanism is unsupported")
    loss_fields = {
        "fault_observed_items",
        "durable_items_at_fault",
        "lost_uncheckpointed_items",
        "replayed_items",
        "final_items",
        "omitted_items",
        "duplicated_committed_items",
    }
    if set(loss) != loss_fields:
        raise RecoveryExperimentError("loss assessment fields do not match schema")
    lost = loss["fault_observed_items"] - loss["durable_items_at_fault"]
    if (
        loss["fault_observed_items"] != fault["fault_after_items"]
        or loss["durable_items_at_fault"] % fault["checkpoint_interval"] != 0
        or lost <= 0
        or loss["lost_uncheckpointed_items"] != lost
        or loss["replayed_items"] != lost
        or loss["final_items"] != experiment["work_items"]
        or loss["omitted_items"] != 0
        or loss["duplicated_committed_items"] != 0
    ):
        raise RecoveryExperimentError("loss assessment does not reconcile")
    timing_fields = {
        "fault_injected_ns",
        "failure_detected_ns",
        "restart_started_ns",
        "recovery_completed_ns",
        "detection_time_ms",
        "restart_delay_ms",
        "recovery_time_ms",
    }
    if set(timing) != timing_fields:
        raise RecoveryExperimentError("timing fields do not match schema")
    recomputed = measure_recovery_timing(
        fault_injected_ns=timing["fault_injected_ns"],
        failure_detected_ns=timing["failure_detected_ns"],
        restart_started_ns=timing["restart_started_ns"],
        recovery_completed_ns=timing["recovery_completed_ns"],
    )
    for field in ("detection_time_ms", "restart_delay_ms", "recovery_time_ms"):
        if timing[field] != getattr(recomputed, field):
            raise RecoveryExperimentError("timing durations do not match boundaries")
    if trace_path is not None:
        trace = load_recovery_trace(trace_path)
        records = json.loads(trace_path.read_bytes())
        if records[-1]["event_sha256"] != report["trace_sha256"]:
            raise RecoveryExperimentError("report trace identity does not match")
        if (
            trace[1].monotonic_ns != timing["fault_injected_ns"]
            or trace[2].monotonic_ns != timing["failure_detected_ns"]
            or trace[3].monotonic_ns != timing["restart_started_ns"]
            or trace[4].monotonic_ns != timing["recovery_completed_ns"]
        ):
            raise RecoveryExperimentError("report timing does not match trace")
    return dict(report)
