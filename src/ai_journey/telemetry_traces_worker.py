"""OpenTelemetry spans from verified worker-recovery evidence.

The worker lab's report is read once as bounded, canonical, duplicate-free
JSON and handed with its trace path to ``verify_recovery_report``. Only after
that verifier accepts it is the trace read once more, re-verified as a hash
chain and bound to the report by its head digest, so the spans describe the
verified records. That verifier never reads the report's evidence boundary,
so the exact text the lab seals is required: a resealed report cannot make
the export claim other evidence. The experiment writes exactly five records
(started, fault_injected, failure_detected, resumed, completed); any other
shape is refused even when the lab verifier tolerates it, and so are payloads
with other keys or types, counts and exit codes that are not exact integers,
and a report whose run, workload, parent checkpoint, recovery time or item
counts differ from its trace. One trace per run holds:

* ``worker-recovery run`` (INTERNAL root) from the start to completion;
* ``worker attempt 1`` (INTERNAL) from the start to the observed failure, in
  ERROR with ``error.type=process_kill``, and ``worker attempt 2`` from the
  resume to completion. Each record becomes an ``ai_journey.lifecycle.<kind>``
  event on its attempt with the record's sequence and hash-chain digest;
* derived phases whose nanosecond durations reproduce the sealed millisecond
  timings: failure detection, worker restart and recovery.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .recovery_trace import RecoveryEvent
from .recovery_trace_verify import verify_recovery_trace
from .recovery_verify import verify_recovery_report
from .telemetry_traces_ids import derive_trace_id, span_context
from .telemetry_traces_io import (
    TelemetryVerificationError,
    canonical_json_bytes,
    read_bounded_file,
    strict_json_loads,
)
from .telemetry_traces_lab import lab_resource
from .telemetry_traces_otlp import (
    MonotonicEpochMapping,
    ResourceTelemetry,
    SpanEvent,
    TelemetrySpan,
)

LAB = "worker-recovery"
SERVICE_NAME = "ai-journey-worker-recovery-lab"
REPORT_FILE = "report.json"
TRACE_FILE = "trace.json"
MAX_REPORT_BYTES = 256 * 1024
MAX_TRACE_BYTES = 1024 * 1024
TRACE_SHAPE = ("started", "fault_injected", "failure_detected", "resumed", "completed")
# The boundary build_recovery_report seals; verify_recovery_report never reads it.
WORKER_EVIDENCE_BOUNDARY = (
    "Measured local CPU process-kill recovery on one host. This does not prove "
    "NVIDIA GPU, distributed, serving, production, model-quality, cost, or "
    "learner-completion evidence."
)
# The sealed timing fields that records 1 to 4 of the trace must carry.
BOUNDARIES = (
    "fault_injected_ns",
    "failure_detected_ns",
    "restart_started_ns",
    "recovery_completed_ns",
)
# Each phase reproduces one sealed duration: name, start, end, milliseconds.
PHASES = (
    (
        "failure detection",
        "fault_injected_ns",
        "failure_detected_ns",
        "detection_time_ms",
    ),
    ("worker restart", "failure_detected_ns", "restart_started_ns", "restart_delay_ms"),
    ("recovery", "fault_injected_ns", "recovery_completed_ns", "recovery_time_ms"),
)
# Report numbers the spans carry; verify_recovery_report compares them by value.
_INTEGERS = {
    "experiment": ("work_items",),
    "fault": ("fault_after_items", "checkpoint_interval"),
    "attempts": ("failed_exit_code", "resumed_exit_code"),
    "loss_assessment": (
        "fault_observed_items",
        "durable_items_at_fault",
        "lost_uncheckpointed_items",
        "replayed_items",
        "final_items",
        "omitted_items",
        "duplicated_committed_items",
    ),
}
# The payload keys and exact types of the five records, in TRACE_SHAPE order.
PAYLOAD_TYPES = (
    {"workload_sha256": str},
    {"mechanism": str},
    {"exit_code": int},
    {"parent_checkpoint_sha256": str},
    {"accumulator_sha256": str, "recovery_time_ms": float},
)
_DIGEST = re.compile(r"[0-9a-f]{64}")
# The float recovery_time_ms payload is carried by the recovery phase span.
_PAYLOAD_ATTRIBUTES = {
    "workload_sha256": "ai_journey.workload.sha256",
    "mechanism": "ai_journey.fault.mechanism",
    "exit_code": "process.exit.code",
    "parent_checkpoint_sha256": "ai_journey.checkpoint.parent_sha256",
    "accumulator_sha256": "ai_journey.accumulator.sha256",
    "recovery_time_ms": None,
}


@dataclass(frozen=True)
class WorkerEvidence:
    """The verified report and the five chained records of one worker run."""

    report: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    events: tuple[RecoveryEvent, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TelemetryVerificationError(message)


def _load_strict(run_dir: Path, name: str, max_bytes: int) -> Any:
    data = read_bounded_file(run_dir / name, max_bytes=max_bytes, label=name)
    value = strict_json_loads(data, label=name)
    _require(canonical_json_bytes(value) == data, f"{name} is not canonical JSON")
    return value


def _require_exact_evidence(
    report: dict[str, Any], events: tuple[RecoveryEvent, ...]
) -> None:
    """Close what the by-value lab verifier leaves open before any span exists.

    ``verify_recovery_report`` accepts ``false`` for ``0`` and ``20.0`` for
    ``20`` and never reads record payloads or the report's run identity, so a
    resealed report could otherwise export boolean counts or name another run.
    """

    _require(
        all(
            type(report[section][name]) is int
            for section, names in _INTEGERS.items()
            for name in names
        ),
        "worker report counts and exit codes must be exact integers",
    )
    for event, types in zip(events, PAYLOAD_TYPES):
        payload = event.payload
        _require(
            set(payload) == set(types)
            and all(type(payload[key]) is kind for key, kind in types.items())
            and all(
                _DIGEST.fullmatch(value) is not None
                for key, value in payload.items()
                if key.endswith("_sha256")
            ),
            f"worker {event.kind} payload differs from the experiment's",
        )
    experiment, attempts, loss = (
        report["experiment"],
        report["attempts"],
        report["loss_assessment"],
    )
    started, injected, detected, resumed, completed = events
    _require(
        experiment["run_id"] == started.run_id
        and started.payload["workload_sha256"] == experiment["workload_sha256"]
        and injected.payload["mechanism"] == report["fault"]["mechanism"]
        and detected.payload["exit_code"] == attempts["failed_exit_code"]
        and resumed.payload["parent_checkpoint_sha256"]
        == attempts["parent_checkpoint_sha256"]
        and completed.payload["recovery_time_ms"]
        == report["timing"]["recovery_time_ms"]
        and [event.completed_items for event in events[1:]]
        == [
            loss["fault_observed_items"],
            loss["fault_observed_items"],
            loss["durable_items_at_fault"],
            loss["final_items"],
        ],
        "worker trace payloads and counts differ from the verified report",
    )


def load_verified_worker_evidence(run_dir: Path) -> WorkerEvidence:
    """Verify a worker run, then load the trace its sealed report binds."""

    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be pathlib.Path")
    try:
        report = _load_strict(run_dir, REPORT_FILE, MAX_REPORT_BYTES)
        verify_recovery_report(report, trace_path=run_dir / TRACE_FILE)
    except (OSError, LookupError, TypeError, ValueError) as error:
        # A short trace makes the lab verifier index past its end.
        raise TelemetryVerificationError(
            f"worker-recovery evidence was rejected: {error}"
        ) from error
    try:
        records = _load_strict(run_dir, TRACE_FILE, MAX_TRACE_BYTES)
        _require(type(records) is list, "worker trace must be a JSON list")
        events = verify_recovery_trace(records)
        _require(
            records[-1]["event_sha256"] == report["trace_sha256"],
            "trace head differs from the verified report",
        )
    except TelemetryVerificationError:
        raise
    except (OSError, LookupError, TypeError, ValueError) as error:
        raise TelemetryVerificationError(
            f"worker-recovery evidence changed after verification: {error}"
        ) from error
    attempts, timing = report["attempts"], report["timing"]
    _require(
        report["evidence_boundary"] == WORKER_EVIDENCE_BOUNDARY,
        "worker report states another evidence boundary than the lab seals",
    )
    _require(
        tuple(event.kind for event in events) == TRACE_SHAPE,
        "worker trace does not have the experiment's five-record shape",
    )
    _require(
        {event.attempt_id for event in events[:3]} == {attempts["failed_attempt_id"]}
        and {event.attempt_id for event in events[3:]}
        == {attempts["resumed_attempt_id"]},
        "worker trace attempts differ from the verified report",
    )
    _require(
        [event.monotonic_ns for event in events[1:]]
        == [timing[name] for name in BOUNDARIES]
        and events[2].payload.get("exit_code") == attempts["failed_exit_code"],
        "worker trace boundaries differ from the verified report",
    )
    _require_exact_evidence(report, events)
    return WorkerEvidence(report, tuple(records), events)


def _lifecycle_events(
    events: tuple[RecoveryEvent, ...],
    records: tuple[dict[str, Any], ...],
    at: Callable[[int], int],
) -> tuple[SpanEvent, ...]:
    converted = []
    for event in events:
        attributes: dict[str, str | bool | int] = {
            "ai_journey.event.sequence": event.sequence,
            "ai_journey.event.sha256": records[event.sequence]["event_sha256"],
            "ai_journey.work.completed_items": event.completed_items,
        }
        for key, value in event.payload.items():
            _require(key in _PAYLOAD_ATTRIBUTES, f"{event.kind} payload has {key}")
            if _PAYLOAD_ATTRIBUTES[key] is not None:
                attributes[_PAYLOAD_ATTRIBUTES[key]] = value
        converted.append(
            SpanEvent(
                f"ai_journey.lifecycle.{event.kind}", at(event.monotonic_ns), attributes
            )
        )
    return tuple(converted)


def _attempt_number(attempt_id: str) -> int:
    return int(attempt_id.split("-")[1])


def worker_resource(
    evidence: WorkerEvidence, *, anchor_unix_nano: int
) -> ResourceTelemetry:
    """Convert verified worker evidence into one labeled resource of spans."""

    if not isinstance(evidence, WorkerEvidence):
        raise TypeError("evidence must be WorkerEvidence")
    report, events, records = evidence.report, evidence.events, evidence.records
    run_id, head = report["experiment"]["run_id"], report["trace_sha256"]
    attempts, loss, timing = (
        report["attempts"],
        report["loss_assessment"],
        report["timing"],
    )
    mapping = MonotonicEpochMapping(
        origin_monotonic_ns=events[0].monotonic_ns, anchor_unix_nano=anchor_unix_nano
    )
    at = mapping.to_unix_nano
    root = span_context(
        derive_trace_id(lab=LAB, run_id=run_id, trace_head_sha256=head), "run"
    )
    spans = [
        TelemetrySpan(
            context=root,
            name="worker-recovery run",
            kind="internal",
            start_unix_nano=at(events[0].monotonic_ns),
            end_unix_nano=at(events[-1].monotonic_ns),
            attributes={
                "ai_journey.run.id": run_id,
                "ai_journey.work.items": report["experiment"]["work_items"],
                "ai_journey.fault.mechanism": report["fault"]["mechanism"],
                "ai_journey.fault.after_items": report["fault"]["fault_after_items"],
                "ai_journey.checkpoint.interval": report["fault"][
                    "checkpoint_interval"
                ],
                "ai_journey.items.lost": loss["lost_uncheckpointed_items"],
                "ai_journey.items.replayed": loss["replayed_items"],
                "ai_journey.items.omitted": loss["omitted_items"],
                "ai_journey.items.duplicated": loss["duplicated_committed_items"],
            },
        )
    ]
    failed, resumed = attempts["failed_attempt_id"], attempts["resumed_attempt_id"]
    spans.append(
        TelemetrySpan(
            context=span_context(
                root.trace_id, f"worker-attempt/{failed}", parent=root
            ),
            name=f"worker attempt {_attempt_number(failed)}",
            kind="internal",
            start_unix_nano=at(events[0].monotonic_ns),
            end_unix_nano=at(events[2].monotonic_ns),
            attributes={
                "ai_journey.attempt.id": failed,
                "ai_journey.attempt.number": _attempt_number(failed),
                "process.exit.code": attempts["failed_exit_code"],
                "error.type": report["fault"]["mechanism"],
            },
            events=_lifecycle_events(events[:3], records, at),
            status="error",
            status_message=(
                f"{report['fault']['mechanism']} after "
                f"{report['fault']['fault_after_items']} completed items"
            ),
        )
    )
    spans.append(
        TelemetrySpan(
            context=span_context(
                root.trace_id, f"worker-attempt/{resumed}", parent=root
            ),
            name=f"worker attempt {_attempt_number(resumed)}",
            kind="internal",
            start_unix_nano=at(events[3].monotonic_ns),
            end_unix_nano=at(events[4].monotonic_ns),
            attributes={
                "ai_journey.attempt.id": resumed,
                "ai_journey.attempt.number": _attempt_number(resumed),
                "ai_journey.attempt.start_items": events[3].completed_items,
                "ai_journey.checkpoint.parent_sha256": attempts[
                    "parent_checkpoint_sha256"
                ],
                "process.exit.code": attempts["resumed_exit_code"],
            },
            events=_lifecycle_events(events[3:], records, at),
        )
    )
    for name, start, end, milliseconds in PHASES:
        _require(
            (timing[end] - timing[start]) / 1_000_000 == timing[milliseconds],
            f"sealed {milliseconds} does not span its boundaries",
        )
        spans.append(
            TelemetrySpan(
                context=span_context(
                    root.trace_id,
                    f"phase/{milliseconds.removesuffix('_ms')}",
                    parent=root,
                ),
                name=name,
                kind="internal",
                start_unix_nano=at(timing[start]),
                end_unix_nano=at(timing[end]),
                attributes={
                    "ai_journey.span.derived": True,
                    "ai_journey.phase.source": f"timing.{milliseconds}",
                },
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
