"""Prometheus metrics derived from one verified worker-recovery run.

The worker report is read as bounded, canonical, strict JSON and handed to
``verify_recovery_report`` with the run's hash-chained trace; only a report
that verifier accepts becomes samples. That verifier binds the report's
timestamps and trace head to the trace but never compares its run identity,
attempt identities, failed exit code or item counts with the events that
record them, so the adapter reads the trace itself and requires each of
those to match. Item accounting restates the loss assessment, and durations
come from the nanosecond boundaries bound to the trace, so the report's
rounded millisecond summary never leaks into the gauges. Every series
carries ``scope="local_cpu_lab"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .recovery_trace import RecoveryEvent
from .recovery_trace_verify import verify_recovery_trace
from .recovery_verify import verify_recovery_report
from .serving_recovery_config import canonical_json_bytes
from .serving_recovery_io import load_canonical_json
from .telemetry_metrics_model import (
    COUNTER,
    GAUGE,
    LAB_SCOPE,
    SCOPE_LABEL,
    MetricFamily,
    TelemetryMetricsError,
    metric_family,
)

REPORT_FILE = "report.json"
TRACE_FILE = "trace.json"
MAX_REPORT_BYTES = 256 * 1024
MAX_TRACE_BYTES = 1024 * 1024
LIFECYCLE = ("started", "fault_injected", "failure_detected", "resumed", "completed")
PREFIX = "ai_journey_worker_recovery_"
_LAB = "local CPU lab telemetry from one verified worker recovery run"
# The worker verifier rejects with ValueError, but some malformed evidence
# makes it fail instead: a zero checkpoint interval divides by zero, a chained
# trace shorter than the lifecycle runs out of events, and a deeply nested
# trace exhausts the JSON decoder. Each of those is rejected evidence too.
_REJECTED = (ArithmeticError, LookupError, RecursionError, TypeError, ValueError)


@dataclass(frozen=True)
class WorkerEvidence:
    """A worker report that ``verify_recovery_report`` accepted with its trace."""

    report: dict[str, Any]


def _traced_fields(
    report: dict[str, Any], events: tuple[RecoveryEvent, ...]
) -> tuple[tuple[str, object, object], ...]:
    """Report fields the worker verifier leaves unbound, with their events."""

    started, fault, detected, resumed, completed = events
    experiment, attempts = report["experiment"], report["attempts"]
    loss = report["loss_assessment"]
    return (
        ("run_id", experiment["run_id"], started.run_id),
        (
            "workload_sha256",
            experiment["workload_sha256"],
            started.payload.get("workload_sha256"),
        ),
        ("failed_attempt_id", attempts["failed_attempt_id"], started.attempt_id),
        (
            "failed_exit_code",
            attempts["failed_exit_code"],
            detected.payload.get("exit_code"),
        ),
        ("resumed_attempt_id", attempts["resumed_attempt_id"], resumed.attempt_id),
        (
            "parent_checkpoint_sha256",
            attempts["parent_checkpoint_sha256"],
            resumed.payload.get("parent_checkpoint_sha256"),
        ),
        ("fault_observed_items", loss["fault_observed_items"], fault.completed_items),
        (
            "fault_observed_items",
            loss["fault_observed_items"],
            detected.completed_items,
        ),
        (
            "durable_items_at_fault",
            loss["durable_items_at_fault"],
            resumed.completed_items,
        ),
        ("final_items", loss["final_items"], completed.completed_items),
    )


def load_verified_worker_evidence(run_dir: Path) -> WorkerEvidence:
    """Load ``report.json`` strictly and verify it against ``trace.json``.

    Raises ``TelemetryMetricsError`` when the report is not canonical strict
    JSON, the worker verifier rejects it or fails on it, or the report and
    its trace disagree. The verifier compares numbers by value, so a resealed
    ``3.0`` would pass it; counts, timestamps and exit codes must also be
    exact integers here. The trace must be the lab's five-event lifecycle,
    re-read under the head digest the verifier accepted, and the run and
    attempt identities, failed exit code and item counts must equal the
    events that record them.
    """

    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be pathlib.Path")
    try:
        report = load_canonical_json(
            run_dir / REPORT_FILE, max_bytes=MAX_REPORT_BYTES, name="worker report"
        )
        verify_recovery_report(report, trace_path=run_dir / TRACE_FILE)
        records = load_canonical_json(
            run_dir / TRACE_FILE, max_bytes=MAX_TRACE_BYTES, name="worker trace"
        )
        events = verify_recovery_trace(records)
    except _REJECTED as error:
        raise TelemetryMetricsError(
            f"worker-recovery evidence was rejected: {error}"
        ) from error
    if records[-1]["event_sha256"] != report["trace_sha256"]:
        raise TelemetryMetricsError("worker trace changed after verification")
    if tuple(event.kind for event in events) != LIFECYCLE:
        raise TelemetryMetricsError("worker trace is not one kill-and-resume lifecycle")
    timing = report["timing"]
    numbers = [
        *report["loss_assessment"].values(),
        *(timing[name] for name in timing if name.endswith("_ns")),
        report["attempts"]["failed_exit_code"],
        report["attempts"]["resumed_exit_code"],
    ]
    if any(type(number) is not int for number in numbers):
        raise TelemetryMetricsError(
            "worker report counts, timestamps and exit codes must be integers"
        )
    for name, stated, traced in _traced_fields(report, events):
        if canonical_json_bytes(stated) != canonical_json_bytes(traced):
            raise TelemetryMetricsError(
                f"worker report {name} does not match its trace"
            )
    return WorkerEvidence(report)


def _scoped(**labels: str) -> dict[str, str]:
    return {**labels, SCOPE_LABEL: LAB_SCOPE}


def worker_recovery_families(evidence: WorkerEvidence) -> tuple[MetricFamily, ...]:
    """Derive every worker metric family from a verified report."""

    if not isinstance(evidence, WorkerEvidence):
        raise TypeError("evidence must be WorkerEvidence")
    report = evidence.report
    loss, timing, attempts = (
        report["loss_assessment"],
        report["timing"],
        report["attempts"],
    )

    def total(name: str, help_text: str, value: int) -> MetricFamily:
        return metric_family(
            PREFIX + name + "_total",
            COUNTER,
            f"{help_text}; {_LAB}.",
            [(_scoped(), value)],
        )

    def gauge(name: str, help_text: str, samples) -> MetricFamily:
        return metric_family(PREFIX + name, GAUGE, f"{help_text}; {_LAB}.", samples)

    def seconds(start: str, end: str) -> list[tuple[dict[str, str], float]]:
        return [(_scoped(), (timing[end] - timing[start]) / 10**9)]

    return (
        total(
            "completed_items",
            "Work items in the final committed result",
            loss["final_items"],
        ),
        total(
            "lost_items",
            "Items completed after the last checkpoint and lost to the kill",
            loss["lost_uncheckpointed_items"],
        ),
        total(
            "replayed_items",
            "Items the resumed attempt recomputed from the last checkpoint",
            loss["replayed_items"],
        ),
        total(
            "omitted_items",
            "Workload items missing from the final result",
            loss["omitted_items"],
        ),
        total(
            "duplicated_items",
            "Items committed more than once in the final result",
            loss["duplicated_committed_items"],
        ),
        gauge(
            "items_at_fault",
            "Items the failed attempt had completed when it was killed",
            [(_scoped(), loss["fault_observed_items"])],
        ),
        gauge(
            "durable_items_at_fault",
            "Items covered by the last checkpoint when the kill landed",
            [(_scoped(), loss["durable_items_at_fault"])],
        ),
        gauge(
            "detection_seconds",
            "Kill issued to the controller detecting the failure",
            seconds("fault_injected_ns", "failure_detected_ns"),
        ),
        gauge(
            "restart_delay_seconds",
            "Failure detection to the resumed attempt starting",
            seconds("failure_detected_ns", "restart_started_ns"),
        ),
        gauge(
            "recovery_seconds",
            "Kill issued to the resumed attempt completing the workload",
            seconds("fault_injected_ns", "recovery_completed_ns"),
        ),
        gauge(
            "attempt_exit_code",
            "Process exit code of the killed and the resumed worker attempt",
            [
                (_scoped(attempt="failed"), attempts["failed_exit_code"]),
                (_scoped(attempt="resumed"), attempts["resumed_exit_code"]),
            ],
        ),
        gauge(
            "run_info",
            "Identity of the verified run these samples were derived from; always 1",
            [
                (
                    _scoped(
                        run_id=report["experiment"]["run_id"],
                        report_sha256=report["report_sha256"],
                        trace_sha256=report["trace_sha256"],
                    ),
                    1,
                )
            ],
        ),
    )
