"""OpenTelemetry spans from verified serving-recovery evidence.

Conversion starts only after ``verify_serving_recovery_run`` accepts the run
directory. The report is then reloaded and must carry the seal the verifier
accepted, and the trace and attempt log are each read once and must hash to
the digests that report binds, so the spans describe exactly the verified
bytes. One trace per run holds:

* ``serving-recovery run`` (INTERNAL root) from the first to the last
  lifecycle record;
* ``server lifetime N`` (INTERNAL) from its spawn or restart to its observed
  exit or stop. Each lifecycle record of the lifetime becomes one span event
  named ``ai_journey.lifecycle.<kind>`` that carries the record's sequence and
  hash-chain digest. The SIGKILLed lifetime ends in ERROR with
  ``error.type=SIGKILL``. ``process.exit.code`` is the status the supervisor
  observed; Python reports death by signal N as -N;
* derived phases (INTERNAL) whose durations equal the sealed timing fields
  they name: fault detection, server restart, client unavailability and SLO
  recovery; a phase whose measurement is null is omitted;
* ``serving request`` (INTERNAL) per logical request, the parent of one
  ``POST /generate`` CLIENT span per attempt with HTTP semantic-convention
  attributes, ``http.request.resend_count`` on retries and ``error.type`` on
  failures (the HTTP status for ``http_error``, otherwise the outcome class).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .serving_recovery_client import AttemptRecord, attempt_from_mapping
from .serving_recovery_experiment import ATTEMPTS_FILE, TRACE_FILE
from .serving_recovery_identity import lifetime_number
from .serving_recovery_io import (
    ATTEMPT_LOG_SCHEMA_VERSION,
    MAX_ATTEMPT_LOG_BYTES,
    MAX_TRACE_BYTES,
)
from .serving_recovery_report import REPORT_FILE, load_serving_report
from .serving_recovery_trace import ServingLifecycleEvent, verify_serving_trace
from .serving_recovery_verify import verify_serving_recovery_run
from .telemetry_traces_ids import derive_trace_id, span_context
from .telemetry_traces_io import (
    TelemetryVerificationError,
    read_bounded_file,
    sha256_hex,
    strict_json_loads,
)
from .telemetry_traces_lab import lab_resource
from .telemetry_traces_otlp import (
    MonotonicEpochMapping,
    ResourceTelemetry,
    SpanEvent,
    TelemetrySpan,
)
from .trace_context import TraceContext

LAB = "serving-recovery"
SERVICE_NAME = "ai-journey-serving-recovery-lab"
LOOPBACK_ADDRESS = "127.0.0.1"
# Each phase reproduces one sealed timing duration: name, start, end, duration.
PHASES = (
    (
        "fault detection",
        "fault_injected_ns",
        "exit_observed_ns",
        "detection_latency_ns",
    ),
    (
        "server restart",
        "exit_observed_ns",
        "replacement_ready_ns",
        "restart_latency_ns",
    ),
    (
        "client unavailability",
        "first_failure_after_fault_ns",
        "first_replacement_success_ns",
        "unavailability_ns",
    ),
    (
        "SLO recovery",
        "fault_injected_ns",
        "slo_recovered_ns",
        "time_to_slo_recovery_ns",
    ),
)
_PAYLOAD_ATTRIBUTES = {
    "pid": "process.pid",
    "port": "server.port",
    "exit_code": "process.exit.code",
    "signal": "ai_journey.fault.signal",
    "acknowledged_responses": "ai_journey.fault.acknowledged_responses",
    "parent_lifetime_id": "ai_journey.lifetime.parent_id",
    "backoff_ms": "ai_journey.restart.backoff_ms",
    "restarts": "ai_journey.restarts",
    "consecutive_successes": "ai_journey.slo.consecutive_successes",
    "slo_latency_ms": "ai_journey.slo.latency_ms",
}


@dataclass(frozen=True)
class ServingEvidence:
    """The verified report, lifecycle records and attempts of one serving run."""

    report: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    events: tuple[ServingLifecycleEvent, ...]
    attempts: tuple[AttemptRecord, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TelemetryVerificationError(message)


def _bound_bytes(run_dir: Path, name: str, max_bytes: int, digest: object) -> bytes:
    data = read_bounded_file(run_dir / name, max_bytes=max_bytes, label=name)
    _require(sha256_hex(data) == digest, f"{name} differs from the verified digest")
    return data


def load_verified_serving_evidence(run_dir: Path) -> ServingEvidence:
    """Verify a serving run, then load exactly the bytes its sealed report binds."""

    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be pathlib.Path")
    try:
        receipt = verify_serving_recovery_run(run_dir)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise TelemetryVerificationError(
            f"serving-recovery evidence was rejected: {error}"
        ) from error
    try:
        report = load_serving_report(run_dir / REPORT_FILE)
        _require(
            report["report_sha256"] == receipt["report_sha256"],
            "report differs from the one the verifier accepted",
        )
        evidence = report["evidence"]
        trace = _bound_bytes(
            run_dir, TRACE_FILE, MAX_TRACE_BYTES, evidence["trace_file_sha256"]
        )
        records = strict_json_loads(trace, label="serving trace")
        _require(type(records) is list, "serving trace must be a JSON list")
        events = verify_serving_trace(records)
        _require(
            records[-1]["event_sha256"] == evidence["trace_head_sha256"],
            "trace head differs from the verified report",
        )
        log = strict_json_loads(
            _bound_bytes(
                run_dir,
                ATTEMPTS_FILE,
                MAX_ATTEMPT_LOG_BYTES,
                evidence["attempts_file_sha256"],
            ),
            label="attempt log",
        )
        _require(
            type(log) is dict
            and set(log) == {"schema_version", "attempts"}
            and log["schema_version"] == ATTEMPT_LOG_SCHEMA_VERSION
            and type(log["attempts"]) is list,
            "attempt log schema differs from the verified version",
        )
        attempts = tuple(attempt_from_mapping(item) for item in log["attempts"])
    except TelemetryVerificationError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise TelemetryVerificationError(
            f"serving-recovery evidence changed after verification: {error}"
        ) from error
    return ServingEvidence(report, tuple(records), events, attempts)


def _lifecycle_event(
    event: ServingLifecycleEvent, record: dict[str, Any], at: Callable[[int], int]
) -> SpanEvent:
    attributes: dict[str, str | bool | int] = {
        "ai_journey.event.sequence": event.sequence,
        "ai_journey.event.sha256": record["event_sha256"],
    }
    for key, value in event.payload.items():
        attributes[_PAYLOAD_ATTRIBUTES[key]] = value
    return SpanEvent(
        f"ai_journey.lifecycle.{event.kind}", at(event.monotonic_ns), attributes
    )


def _lifetime_span(
    lifetime_id: str,
    own: list[ServingLifecycleEvent],
    records: tuple[dict[str, Any], ...],
    root: TraceContext,
    at: Callable[[int], int],
) -> TelemetrySpan:
    begin, end = own[0], own[-1]
    _require(
        begin.kind in ("spawn", "restart")
        and end.kind in ("exit_observed", "finished"),
        "a lifetime does not run from its start to an observed end",
    )
    attributes: dict[str, str | bool | int] = {
        "ai_journey.lifetime.id": lifetime_id,
        "ai_journey.lifetime.number": lifetime_number(lifetime_id),
        "process.pid": begin.payload["pid"],
        "process.exit.code": end.payload["exit_code"],
    }
    ready = [event for event in own if event.kind == "ready"]
    if ready:
        attributes["server.address"] = LOOPBACK_ADDRESS
        attributes["server.port"] = ready[0].payload["port"]
    if begin.kind == "restart":
        attributes["ai_journey.lifetime.parent_id"] = begin.payload[
            "parent_lifetime_id"
        ]
        attributes["ai_journey.restart.backoff_ms"] = begin.payload["backoff_ms"]
    faults = [event for event in own if event.kind == "fault"]
    status, message = "unset", None
    if faults:
        attributes["error.type"] = faults[0].payload["signal"]
        status = "error"
        message = (
            f"{faults[0].payload['signal']} injected after "
            f"{faults[0].payload['acknowledged_responses']} acknowledged responses"
        )
    elif end.kind == "exit_observed":
        hung = any(event.kind == "unresponsive" for event in own)
        attributes["error.type"] = "unresponsive" if hung else "unexpected_exit"
        status, message = "error", "the lifetime exited before the run finished"
    return TelemetrySpan(
        context=span_context(root.trace_id, f"lifetime/{lifetime_id}", parent=root),
        name=f"server lifetime {lifetime_number(lifetime_id)}",
        kind="internal",
        start_unix_nano=at(begin.monotonic_ns),
        end_unix_nano=at(end.monotonic_ns),
        attributes=attributes,
        events=tuple(_lifecycle_event(e, records[e.sequence], at) for e in own),
        status=status,
        status_message=message,
    )


def _phase_spans(
    timing: dict[str, Any], root: TraceContext, at: Callable[[int], int]
) -> list[TelemetrySpan]:
    spans = []
    for name, start, end, duration in PHASES:
        if timing[duration] is None:
            continue
        _require(
            timing[end] - timing[start] == timing[duration],
            f"sealed {duration} does not span its boundaries",
        )
        spans.append(
            TelemetrySpan(
                context=span_context(
                    root.trace_id, f"phase/{duration.removesuffix('_ns')}", parent=root
                ),
                name=name,
                kind="internal",
                start_unix_nano=at(timing[start]),
                end_unix_nano=at(timing[end]),
                attributes={
                    "ai_journey.span.derived": True,
                    "ai_journey.phase.source": f"timing.{duration}",
                },
            )
        )
    return spans


def _attempt_span(
    attempt: AttemptRecord, parent: TraceContext, at: Callable[[int], int]
) -> TelemetrySpan:
    attributes: dict[str, str | bool | int] = {
        "http.request.method": "POST",
        "url.full": f"http://{LOOPBACK_ADDRESS}:{attempt.port}/generate",
        "url.template": "/generate",
        "server.address": LOOPBACK_ADDRESS,
        "server.port": attempt.port,
        "network.protocol.version": "1.0",
        "ai_journey.attempt.number": attempt.attempt,
        "ai_journey.attempt.outcome": attempt.outcome,
        "ai_journey.attempt.connected": attempt.connected,
    }
    if attempt.attempt > 1:
        attributes["http.request.resend_count"] = attempt.attempt - 1
    if attempt.http_status is not None:
        attributes["http.response.status_code"] = attempt.http_status
    if attempt.lifetime_id is not None:
        attributes["ai_journey.lifetime.id"] = attempt.lifetime_id
    status, message = "unset", None
    if attempt.outcome != "ok":
        attributes["ai_journey.attempt.error"] = attempt.error
        attributes["error.type"] = (
            str(attempt.http_status)
            if attempt.outcome == "http_error"
            else attempt.outcome
        )
        status, message = "error", f"{attempt.outcome}: {attempt.error}"
    return TelemetrySpan(
        context=span_context(
            parent.trace_id,
            f"attempt/{attempt.idempotency_key}/{attempt.attempt}",
            parent=parent,
        ),
        name="POST /generate",
        kind="client",
        start_unix_nano=at(attempt.started_ns),
        end_unix_nano=at(attempt.finished_ns),
        attributes=attributes,
        status=status,
        status_message=message,
    )


def _request_spans(
    requests: list[dict[str, Any]],
    attempts: tuple[AttemptRecord, ...],
    root: TraceContext,
    at: Callable[[int], int],
) -> list[TelemetrySpan]:
    by_key: dict[str, list[AttemptRecord]] = {}
    for attempt in attempts:
        by_key.setdefault(attempt.idempotency_key, []).append(attempt)
    _require(
        sorted(by_key) == sorted(entry["idempotency_key"] for entry in requests),
        "attempts name requests outside the sealed reconciliation",
    )
    spans = []
    for entry in requests:
        key = entry["idempotency_key"]
        own = sorted(by_key[key], key=lambda a: a.attempt)
        _require(
            len(own) == entry["attempts"],
            "a request's attempts differ from its sealed reconciliation",
        )
        context = span_context(root.trace_id, f"request/{key}", parent=root)
        failed = entry["outcome"] == "failed_after_budget"
        spans.append(
            TelemetrySpan(
                context=context,
                name="serving request",
                kind="internal",
                start_unix_nano=at(min(a.started_ns for a in own)),
                end_unix_nano=at(max(a.finished_ns for a in own)),
                attributes={
                    "ai_journey.request.index": entry["request_index"],
                    "ai_journey.request.idempotency_key": key,
                    "ai_journey.request.outcome": entry["outcome"],
                    "ai_journey.request.attempts": entry["attempts"],
                    "ai_journey.request.executions": entry["executions"],
                    "ai_journey.request.lost_response_executions": entry[
                        "lost_response_executions"
                    ],
                },
                status="error" if failed else "unset",
                status_message="the retry budget ran out" if failed else None,
            )
        )
        spans.extend(_attempt_span(attempt, context, at) for attempt in own)
    return spans


def serving_resource(
    evidence: ServingEvidence, *, anchor_unix_nano: int
) -> ResourceTelemetry:
    """Convert verified serving evidence into one labeled resource of spans."""

    if not isinstance(evidence, ServingEvidence):
        raise TypeError("evidence must be ServingEvidence")
    report, events = evidence.report, evidence.events
    run_id = report["experiment"]["run_id"]
    head = report["evidence"]["trace_head_sha256"]
    mapping = MonotonicEpochMapping(
        origin_monotonic_ns=events[0].monotonic_ns, anchor_unix_nano=anchor_unix_nano
    )
    at = mapping.to_unix_nano
    root = span_context(
        derive_trace_id(lab=LAB, run_id=run_id, trace_head_sha256=head), "run"
    )
    totals = report["reconciliation"]["totals"]
    spans = [
        TelemetrySpan(
            context=root,
            name="serving-recovery run",
            kind="internal",
            start_unix_nano=at(events[0].monotonic_ns),
            end_unix_nano=at(events[-1].monotonic_ns),
            attributes={
                "ai_journey.run.id": run_id,
                "ai_journey.recovered": report["recovered"],
                "ai_journey.fault.mechanism": report["fault"]["mechanism"],
                "ai_journey.requests": totals["requests"],
                "ai_journey.attempts": totals["attempts"],
                "ai_journey.executions": totals["executions"],
                "ai_journey.duplicate_executions": totals["duplicate_executions"],
                "ai_journey.slo.latency_ms": report["slo"]["latency_ms"],
                "ai_journey.slo.consecutive_successes": report["slo"][
                    "consecutive_successes"
                ],
            },
            status="unset" if report["recovered"] else "error",
            status_message=(
                None if report["recovered"] else "the run did not recover in bounds"
            ),
        )
    ]
    lifetimes = [e.lifetime_id for e in events if e.kind in ("spawn", "restart")]
    for lifetime_id in lifetimes:
        own = [event for event in events if event.lifetime_id == lifetime_id]
        spans.append(_lifetime_span(lifetime_id, own, evidence.records, root, at))
    spans.extend(_phase_spans(report["timing"], root, at))
    spans.extend(
        _request_spans(
            report["reconciliation"]["requests"], evidence.attempts, root, at
        )
    )
    return lab_resource(
        lab=LAB,
        service_name=SERVICE_NAME,
        run_id=run_id,
        report_sha256=report["report_sha256"],
        trace_head_sha256=head,
        lab_boundary=report["evidence_boundary"],
        mapping=mapping,
        spans=tuple(spans),
    )
