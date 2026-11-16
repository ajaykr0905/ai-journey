"""Independently verify a sealed trace export against its source evidence.

The verifier trusts no stored summary. It checks the report's seal, schema,
scope, boundary, encoding statement and embedded config; re-reads the export
bytes, their size and digest; and validates their structure. It then re-runs
each source lab's verifier through the export pipeline, rebuilds the export
from the raw evidence under the embedded config, and requires the same bytes
and source statements, then the same report bytes, so no sealed field can
hold another JSON spelling of its value, such as ``1.0`` or ``true`` for
``1``. Separately from the encoder, it reads the OTLP JSON directly and maps
it back to the evidence: one event per lifecycle record carrying that
record's sequence, digest and mapped time; resource identities and clock
mappings equal to the run's; derived phase durations equal to the sealed
timings; and, for the serving lab, one CLIENT span per recorded attempt,
found through its parent request's idempotency key and its attempt number,
with that attempt's mapped start and finish, outcome and error status.

A report that cannot be read raises ``TelemetryError``; anything read and
then found wrong raises ``TelemetryVerificationError``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .telemetry_traces_config import config_from_mapping, load_telemetry_config
from .telemetry_traces_io import (
    TelemetryError,
    TelemetryVerificationError,
    artifact_bytes,
    read_bounded_file,
    sha256_hex,
    strict_json_loads,
)
from .telemetry_traces_lab import TELEMETRY_BOUNDARY
from .telemetry_traces_report import (
    ENCODING,
    EXPORT_FILE,
    MAX_EXPORT_BYTES,
    MAX_REPORT_BYTES,
    REPORT_KIND,
    REPORT_SCHEMA_VERSION,
    build_trace_export,
    build_trace_export_report,
    require_sealed_report,
)
from .telemetry_traces_serving import PHASES as SERVING_PHASES
from .telemetry_traces_serving import ServingEvidence
from .telemetry_traces_validate import LOCAL_CPU_LAB, validate_export_request
from .telemetry_traces_worker import PHASES as WORKER_PHASES

_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "telemetry_scope",
        "evidence_boundary",
        "config",
        "config_sha256",
        "encoding",
        "sources",
        "export",
        "report_sha256",
    }
)
_CLIENT = 3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TelemetryVerificationError(message)


def _decode(attributes: list[dict[str, Any]]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for entry in attributes:
        ((kind, value),) = entry["value"].items()
        decoded[entry["key"]] = int(value) if kind == "intValue" else value
    return decoded


def _check_events(spans: list[dict[str, Any]], records, origin: int, anchor: int):
    seen: dict[int, dict[str, Any]] = {}
    for span in spans:
        for event in span.get("events", []):
            attributes = _decode(event.get("attributes", []))
            sequence = attributes.get("ai_journey.event.sequence")
            _require(
                type(sequence) is int and sequence not in seen,
                "a lifecycle event is missing or repeats its sequence",
            )
            seen[sequence] = event | {"decoded": attributes}
    _require(
        sorted(seen) == list(range(len(records))),
        "lifecycle events do not cover every trace record exactly once",
    )
    for sequence, event in seen.items():
        record = records[sequence]
        _require(
            event["decoded"].get("ai_journey.event.sha256") == record["event_sha256"]
            and event["name"] == f"ai_journey.lifecycle.{record['kind']}"
            and int(event["timeUnixNano"]) == anchor + record["monotonic_ns"] - origin,
            f"lifecycle event {sequence} does not map its trace record",
        )


def _check_phases(
    spans: list[dict[str, Any]], evidence, origin: int, anchor: int
) -> None:
    timing = evidence.report["timing"]
    serving = isinstance(evidence, ServingEvidence)
    expected = {
        f"timing.{duration}": (start, duration)
        for _, start, _, duration in (SERVING_PHASES if serving else WORKER_PHASES)
        if timing[duration] is not None
    }
    phases = [
        span
        for span in spans
        if _decode(span["attributes"]).get("ai_journey.span.derived") is True
    ]
    sources = [
        _decode(span["attributes"]).get("ai_journey.phase.source") for span in phases
    ]
    _require(
        sorted(map(str, sources)) == sorted(expected),
        "derived phases differ from the sealed timing fields",
    )
    for source, span in zip(sources, phases):
        start, field = expected[source]
        begin = int(span["startTimeUnixNano"])
        duration = int(span["endTimeUnixNano"]) - begin
        # Serving timings are sealed in nanoseconds, worker timings in ms.
        if not serving:
            duration = duration / 1_000_000
        _require(
            begin == anchor + timing[start] - origin and duration == timing[field],
            f"phase {source} does not span its sealed boundaries",
        )


def _check_attempts(
    spans: list[dict[str, Any]], evidence: ServingEvidence, origin: int, anchor: int
) -> None:
    clients = [span for span in spans if span["kind"] == _CLIENT]
    attempts = evidence.attempts
    _require(len(clients) == len(attempts), "CLIENT spans differ from the attempts")
    _require(
        sorted(int(s["endTimeUnixNano"]) - int(s["startTimeUnixNano"]) for s in clients)
        == sorted(attempt.latency_ns for attempt in attempts),
        "CLIENT span durations differ from the attempt latencies",
    )
    _require(
        sum(span["status"]["code"] == 2 for span in clients)
        == sum(attempt.outcome != "ok" for attempt in attempts),
        "CLIENT span errors differ from the failed attempts",
    )
    # Multisets miss an attempt parented under another request, so map each.
    parents = {span["spanId"]: span for span in spans}
    records = {(a.idempotency_key, a.attempt): a for a in attempts}
    for span in clients:
        parent = parents.get(span.get("parentSpanId"), {"attributes": []})
        key = _decode(parent["attributes"]).get("ai_journey.request.idempotency_key")
        decoded = _decode(span["attributes"])
        attempt = records.pop((key, decoded.get("ai_journey.attempt.number")), None)
        _require(
            attempt is not None
            and int(span["startTimeUnixNano"]) == anchor + attempt.started_ns - origin
            and int(span["endTimeUnixNano"]) == anchor + attempt.finished_ns - origin
            and decoded.get("ai_journey.attempt.outcome") == attempt.outcome
            and (span["status"]["code"] == 2) is (attempt.outcome != "ok"),
            "a CLIENT span does not map its attempt record",
        )


def _check_mapping(request: dict[str, Any], export, anchor: int) -> None:
    resources = request["resourceSpans"]
    _require(len(resources) == len(export.evidence), "export resources differ")
    for resource, evidence in zip(resources, export.evidence):
        attributes = _decode(resource["resource"]["attributes"])
        spans = resource["scopeSpans"][0]["spans"]
        records = evidence.records
        origin = records[0]["monotonic_ns"]
        _require(
            attributes.get("ai_journey.run.id")
            == evidence.report["experiment"]["run_id"]
            and attributes.get("ai_journey.evidence.report_sha256")
            == evidence.report["report_sha256"]
            and attributes.get("ai_journey.clock.origin_monotonic_ns") == origin
            and attributes.get("ai_journey.clock.anchor_unix_nano") == anchor,
            "a resource does not name its run, report and clock mapping",
        )
        _check_events(spans, records, origin, anchor)
        _check_phases(spans, evidence, origin, anchor)
        if isinstance(evidence, ServingEvidence):
            _check_attempts(spans, evidence, origin, anchor)


def verify_trace_export(
    report_path: Path,
    *,
    serving_run_dir: Path | None = None,
    worker_run_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Recompute a sealed export from its sources and return a receipt.

    The run directory of every exported lab is required, and no other. A
    ``config_path`` additionally binds the embedded config to a pinned file.
    """

    data = read_bounded_file(
        report_path, max_bytes=MAX_REPORT_BYTES, label="trace export report"
    )
    pinned = load_telemetry_config(config_path) if config_path is not None else None
    try:
        report = strict_json_loads(data, label="trace export report")
        _require(artifact_bytes(report) == data, "report is not canonical JSON")
        require_sealed_report(report, kind=REPORT_KIND)
        # A sealed report may still lack a field, so check before reading one.
        _require(set(report) == _REPORT_FIELDS, "report fields do not match schema")
        config = config_from_mapping(report["config"])
    except TelemetryVerificationError:
        raise
    except TelemetryError as error:
        raise TelemetryVerificationError(str(error)) from error
    _require(
        report["schema_version"] == REPORT_SCHEMA_VERSION,
        "report schema is unsupported",
    )
    _require(report["telemetry_scope"] == LOCAL_CPU_LAB, "telemetry scope changed")
    _require(report["evidence_boundary"] == TELEMETRY_BOUNDARY, "boundary changed")
    _require(report["encoding"] == ENCODING, "encoding statement changed")
    _require(report["config_sha256"] == config.fingerprint(), "config digest differs")
    _require(
        pinned is None or pinned == config,
        "report was built from another config than the pinned one",
    )
    exported = report["sources"]
    _require(
        type(exported) is list
        and bool(exported)
        and all(type(source) is dict for source in exported),
        "report sources must be a non-empty list of objects",
    )
    labs = [source.get("lab") for source in exported]
    given = [
        lab
        for lab, path in (
            ("serving-recovery", serving_run_dir),
            ("worker-recovery", worker_run_dir),
        )
        if path is not None
    ]
    if labs != given:
        raise TelemetryError(
            "verify needs the run directory of exactly the exported labs: "
            + ", ".join(str(lab) for lab in labs)
        )
    stored = report["export"]
    _require(
        type(stored) is dict and stored.get("file") == EXPORT_FILE,
        "report names an unexpected export file",
    )
    try:
        export_bytes = read_bounded_file(
            report_path.parent / EXPORT_FILE, max_bytes=MAX_EXPORT_BYTES, label="export"
        )
        request = strict_json_loads(export_bytes, label="export")
    except TelemetryError as error:
        raise TelemetryVerificationError(str(error)) from error
    _require(
        stored.get("bytes") == len(export_bytes)
        and stored.get("sha256") == sha256_hex(export_bytes),
        "export bytes do not match the sealed digest",
    )
    _require(artifact_bytes(request) == export_bytes, "export is not canonical JSON")
    summary = validate_export_request(request)
    _require(
        {key: stored[key] for key in stored if key not in ("file", "bytes", "sha256")}
        == summary.to_mapping(),
        "export summary does not match its contents",
    )
    rebuilt = build_trace_export(
        config, serving_run_dir=serving_run_dir, worker_run_dir=worker_run_dir
    )
    _require(
        rebuilt.data == export_bytes,
        "export does not recompute from the source evidence",
    )
    _require(
        list(rebuilt.sources) == exported,
        "report sources do not recompute from the evidence",
    )
    # Python equality calls true, 1 and 1.0 equal, so compare canonical bytes.
    _require(
        artifact_bytes(build_trace_export_report(config, rebuilt)) == data,
        "report does not recompute byte for byte from the source evidence",
    )
    _check_mapping(request, rebuilt, config.anchor_unix_nano)
    return {
        "verified": True,
        "kind": REPORT_KIND,
        "telemetry_scope": LOCAL_CPU_LAB,
        "report_sha256": report["report_sha256"],
        "export_sha256": stored["sha256"],
        "spans": summary.spans,
        "traces": [
            {
                "lab": source["lab"],
                "run_id": source["run_id"],
                "traceparent": source["traceparent"],
            }
            for source in exported
        ],
    }
