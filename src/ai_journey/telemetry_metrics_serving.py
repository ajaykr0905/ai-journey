"""Prometheus metrics derived from one verified serving-recovery run.

Nothing is derived until ``verify_serving_recovery_run`` has recomputed the
run from its raw evidence and accepted it. The adapter then loads the sealed
report and the attempt log and binds both to that verification: the report
by its seal, and the parsed attempts by re-encoding them to the digest the
verified report records, so a file replaced after verification is refused
rather than exported. Counts come from the verified reconciliation,
durations from the verified nanosecond timing, and the latency histograms
from the attempt log, cross-checked against the report's latency sample
counts. Every series carries ``scope="local_cpu_lab"``.

Success latencies fall into three phases that partition the acknowledged
attempts exactly as the serving lab's timing does: ``pre_fault`` finished by
the kill, ``late_acknowledgement`` was sent by the killed lifetime but
arrived after the kill, and ``post_fault`` was served by a later lifetime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .serving_recovery_client import ATTEMPT_OUTCOMES, AttemptRecord
from .serving_recovery_config import canonical_json_bytes, sha256_hex
from .serving_recovery_experiment import ATTEMPTS_FILE
from .serving_recovery_identity import lifetime_number
from .serving_recovery_io import ATTEMPT_LOG_SCHEMA_VERSION, load_attempt_log
from .serving_recovery_reconcile import REQUEST_OUTCOMES
from .serving_recovery_report import REPORT_FILE, load_serving_report
from .serving_recovery_verify import verify_serving_recovery_run
from .telemetry_metrics_model import (
    COUNTER,
    GAUGE,
    HISTOGRAM,
    LAB_SCOPE,
    SCOPE_LABEL,
    MetricFamily,
    TelemetryMetricsError,
    integer_histogram,
    metric_family,
    scaled_bounds,
)

PREFIX = "ai_journey_serving_recovery_"
PHASES = ("pre_fault", "late_acknowledgement", "post_fault")
FAILED_OUTCOMES = tuple(outcome for outcome in ATTEMPT_OUTCOMES if outcome != "ok")
_LAB = "local CPU lab telemetry from one verified serving recovery run"


@dataclass(frozen=True)
class ServingEvidence:
    """The verifier's receipt with the sealed report and attempts it accepted."""

    receipt: dict[str, Any]
    report: dict[str, Any]
    attempts: tuple[AttemptRecord, ...]


def load_verified_serving_evidence(run_dir: Path) -> ServingEvidence:
    """Verify a serving run directory, then load what was verified.

    Raises ``TelemetryMetricsError`` when the serving verifier rejects the
    evidence or a file no longer matches what it accepted.
    """

    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be pathlib.Path")
    try:
        receipt = verify_serving_recovery_run(run_dir)
        report = load_serving_report(run_dir / REPORT_FILE)
        attempts = load_attempt_log(run_dir / ATTEMPTS_FILE)
    except (KeyError, TypeError, ValueError) as error:
        raise TelemetryMetricsError(
            f"serving-recovery evidence was rejected: {error}"
        ) from error
    if report["report_sha256"] != receipt["report_sha256"]:
        raise TelemetryMetricsError("serving report changed after verification")
    encoded = canonical_json_bytes(
        {
            "schema_version": ATTEMPT_LOG_SCHEMA_VERSION,
            "attempts": [attempt.to_mapping() for attempt in attempts],
        }
    )
    if sha256_hex(encoded) != report["evidence"]["attempts_file_sha256"]:
        raise TelemetryMetricsError("serving attempt log changed after verification")
    return ServingEvidence(receipt, report, attempts)


def _scoped(**labels: str) -> dict[str, str]:
    return {**labels, SCOPE_LABEL: LAB_SCOPE}


def _seconds(nanoseconds: int | None) -> list[tuple[dict[str, str], float]]:
    return [] if nanoseconds is None else [(_scoped(), nanoseconds / 10**9)]


def success_phase(attempt: AttemptRecord, *, fault_ns: int, faulted: int) -> str:
    """Name the latency phase of one acknowledged attempt."""

    if lifetime_number(attempt.lifetime_id) > faulted:
        return "post_fault"
    return "pre_fault" if attempt.finished_ns <= fault_ns else "late_acknowledgement"


def _latency_families(
    evidence: ServingEvidence, latency_buckets_us: Sequence[int]
) -> list[MetricFamily]:
    report, attempts = evidence.report, evidence.attempts
    fault_ns = report["fault"]["injected_ns"]
    faulted = lifetime_number(report["fault"]["lifetime_id"])
    phases: dict[str, list[int]] = {phase: [] for phase in PHASES}
    for attempt in attempts:
        if attempt.outcome == "ok":
            phase = success_phase(attempt, fault_ns=fault_ns, faulted=faulted)
            phases[phase].append(attempt.latency_ns)
    timing = report["timing"]
    expected = (
        timing["pre_fault_latency"]["samples"],
        timing["late_acknowledgements"],
        timing["post_fault_latency"]["samples"],
    )
    if tuple(len(phases[phase]) for phase in PHASES) != expected:
        raise TelemetryMetricsError("latency phases do not match the verified timing")
    bounds_ns = [bound * 1000 for bound in latency_buckets_us]
    buckets = scaled_bounds(latency_buckets_us, scale=10**6)

    def histogram(values: list[int]):
        return integer_histogram(values, bounds=bounds_ns, scale=10**9)

    return [
        metric_family(
            PREFIX + "success_latency_seconds",
            HISTOGRAM,
            "Acknowledged attempt latency by recovery phase, start to finish on "
            f"the experiment host's monotonic clock; {_LAB}.",
            [(_scoped(phase=phase), histogram(phases[phase])) for phase in PHASES],
            buckets=buckets,
        ),
        metric_family(
            PREFIX + "failed_attempt_duration_seconds",
            HISTOGRAM,
            f"Duration of failed client attempts by outcome class; {_LAB}.",
            [
                (
                    _scoped(outcome=outcome),
                    histogram([a.latency_ns for a in attempts if a.outcome == outcome]),
                )
                for outcome in FAILED_OUTCOMES
            ],
            buckets=buckets,
        ),
    ]


def serving_recovery_families(
    evidence: ServingEvidence, *, latency_buckets_us: Sequence[int]
) -> tuple[MetricFamily, ...]:
    """Derive every serving metric family from verified evidence."""

    if not isinstance(evidence, ServingEvidence):
        raise TypeError("evidence must be ServingEvidence")
    report = evidence.report
    reconciliation = report["reconciliation"]
    totals, timing = reconciliation["totals"], report["timing"]
    outcomes = reconciliation["attempt_outcomes"]

    def total(name: str, help_text: str, value: int) -> MetricFamily:
        return metric_family(
            PREFIX + name + "_total",
            COUNTER,
            f"{help_text}; {_LAB}.",
            [(_scoped(), value)],
        )

    def gauge(name: str, help_text: str, samples) -> MetricFamily:
        return metric_family(PREFIX + name, GAUGE, f"{help_text}; {_LAB}.", samples)

    return (
        metric_family(
            PREFIX + "attempts_total",
            COUNTER,
            f"Client attempts by outcome class; {_LAB}.",
            [(_scoped(outcome=name), outcomes[name]) for name in ATTEMPT_OUTCOMES],
        ),
        metric_family(
            PREFIX + "requests_total",
            COUNTER,
            f"Logical requests by reconciled final outcome; {_LAB}.",
            [(_scoped(outcome=name), totals[name]) for name in REQUEST_OUTCOMES],
        ),
        metric_family(
            PREFIX + "executions_total",
            COUNTER,
            f"Committed journal executions by server lifetime number; {_LAB}.",
            [
                (_scoped(lifetime=str(lifetime_number(lifetime))), count)
                for lifetime, count in reconciliation["executions_by_lifetime"].items()
            ],
        ),
        total(
            "duplicate_executions",
            "Executions beyond the first of one idempotency key, the at-least-once "
            "duplicates a retry after a lost response causes",
            totals["duplicate_executions"],
        ),
        total(
            "lost_response_executions",
            "Executions whose response never reached the client",
            totals["lost_response_executions"],
        ),
        total(
            "retried_attempts",
            "Attempts after the first of each logical request",
            totals["retried_attempts"],
        ),
        total(
            "late_acknowledgements",
            "Successes the killed lifetime sent that arrived after the kill",
            timing["late_acknowledgements"],
        ),
        total(
            "restarts",
            "Supervisor restarts that started a replacement lifetime",
            len(report["experiment"]["lifetimes"]) - 1,
        ),
        gauge(
            "detection_latency_seconds",
            "Kill issued to the supervisor observing the exit",
            _seconds(timing["detection_latency_ns"]),
        ),
        gauge(
            "restart_latency_seconds",
            "Observed exit to replacement readiness, including the restart backoff",
            _seconds(timing["restart_latency_ns"]),
        ),
        gauge(
            "unavailability_seconds",
            "First failed attempt after the kill to the first post-fault success; "
            "no sample when no client failed first",
            _seconds(timing["unavailability_ns"]),
        ),
        gauge(
            "time_to_slo_recovery_seconds",
            "Kill to the end of the first run of consecutive in-SLO successes; "
            "no sample when that run was never observed",
            _seconds(timing["time_to_slo_recovery_ns"]),
        ),
        gauge(
            "recovered",
            "1 when every request succeeded and SLO recovery was observed, else 0",
            [(_scoped(), int(report["recovered"]))],
        ),
        gauge(
            "run_info",
            "Identity of the verified run these samples were derived from; always 1",
            [
                (
                    _scoped(
                        run_id=report["experiment"]["run_id"],
                        report_sha256=report["report_sha256"],
                    ),
                    1,
                )
            ],
        ),
        *_latency_families(evidence, latency_buckets_us),
    )
