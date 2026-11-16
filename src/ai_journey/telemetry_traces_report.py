"""Export verified lab traces as OTLP/JSON with a sealed report.

``build_trace_export`` converts each given run directory only after its lab
verifier accepts it, encodes one ExportTraceServiceRequest with one resource
per run, validates the encoded JSON and renders canonical bytes. The sealed
report binds those bytes by SHA-256 to the config, the encoding rules, the
time mapping and the source evidence of every run (report seal and trace-head
digest, never a filesystem path), and states that this is local CPU lab
telemetry. The seal is the SHA-256 of the canonical report without the seal
field: it proves content identity, not authorship. ``telemetry_traces_verify``
recomputes the export from the source runs instead of trusting this summary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .telemetry_traces_config import TelemetryConfig
from .telemetry_traces_io import (
    TelemetryError,
    artifact_bytes,
    canonical_sha256,
    create_output_directory,
    load_canonical_artifact,
    publish_new_file,
    sha256_hex,
)
from .telemetry_traces_lab import TELEMETRY_BOUNDARY
from .telemetry_traces_otlp import (
    MonotonicEpochMapping,
    ResourceTelemetry,
    encode_export_request,
)
from .telemetry_traces_serving import (
    ServingEvidence,
    load_verified_serving_evidence,
    serving_resource,
)
from .telemetry_traces_validate import (
    LOCAL_CPU_LAB,
    ExportSummary,
    validate_export_request,
)
from .telemetry_traces_worker import (
    WorkerEvidence,
    load_verified_worker_evidence,
    worker_resource,
)
from .trace_context import TraceContext

EXPORT_FILE = "traces.otlp.json"
REPORT_FILE = "report.json"
REPORT_KIND = "ai-journey-telemetry-trace-export"
REPORT_SCHEMA_VERSION = 1
MAX_EXPORT_BYTES = 32 * 1024 * 1024
MAX_REPORT_BYTES = 256 * 1024
ENCODING = {
    "format": "OTLP/JSON ExportTraceServiceRequest (opentelemetry-proto trace/v1)",
    "ids": "trace and span IDs as lowercase hexadecimal strings",
    "times": "unsigned 64-bit Unix nanoseconds as decimal strings",
    "enums": "SpanKind and StatusCode as integers",
    "attribute_values": "stringValue, boolValue, or intValue as a decimal string",
    "span_order": "start time, then depth, so each parent precedes its children",
}


def seal_report(report: dict[str, Any]) -> dict[str, Any]:
    """Add the canonical SHA-256 seal over every other report field."""

    if type(report) is not dict or "report_sha256" in report:
        raise TelemetryError("report must be an unsealed mapping")
    return {**report, "report_sha256": canonical_sha256(report)}


def require_sealed_report(report: object, *, kind: str) -> dict[str, Any]:
    """Return a report of ``kind`` whose seal matches its canonical content."""

    if type(report) is not dict or type(report.get("report_sha256")) is not str:
        raise TelemetryError("report is not sealed")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if canonical_sha256(unsigned) != report["report_sha256"]:
        raise TelemetryError("report seal does not match its content")
    if report.get("kind") != kind:
        raise TelemetryError(f"report kind is not {kind}")
    return report


@dataclass(frozen=True)
class TraceExport:
    """Encoded, validated export bytes and the verified runs they came from.

    ``evidence`` holds each run's verified evidence in the order of
    ``sources`` so a verifier can map spans back without re-verifying runs.
    """

    request: dict[str, Any]
    data: bytes
    sources: tuple[dict[str, Any], ...]
    summary: ExportSummary
    evidence: tuple[ServingEvidence | WorkerEvidence, ...]


def _source(lab: str, resource: ResourceTelemetry) -> dict[str, Any]:
    attributes = resource.attributes
    (root,) = [span for span in resource.spans if span.parent_span_id is None]
    mapping = MonotonicEpochMapping(
        origin_monotonic_ns=attributes["ai_journey.clock.origin_monotonic_ns"],
        anchor_unix_nano=attributes["ai_journey.clock.anchor_unix_nano"],
    )
    return {
        "lab": lab,
        "service_name": attributes["service.name"],
        "run_id": attributes["ai_journey.run.id"],
        "report_sha256": attributes["ai_journey.evidence.report_sha256"],
        "trace_head_sha256": attributes["ai_journey.evidence.trace_head_sha256"],
        "evidence_boundary": attributes["ai_journey.evidence.lab_boundary"],
        "trace_id": root.trace_id,
        "root_span_id": root.span_id,
        "traceparent": TraceContext(root.trace_id, root.span_id, True).to_header(),
        "time_mapping": mapping.to_mapping(),
        "spans": len(resource.spans),
    }


def build_trace_export(
    config: TelemetryConfig,
    *,
    serving_run_dir: Path | None = None,
    worker_run_dir: Path | None = None,
) -> TraceExport:
    """Verify each given run, convert it to spans and encode one export."""

    if not isinstance(config, TelemetryConfig):
        raise TypeError("config must be TelemetryConfig")
    anchor = config.anchor_unix_nano
    runs: list[tuple[str, ServingEvidence | WorkerEvidence, ResourceTelemetry]] = []
    if serving_run_dir is not None:
        serving = load_verified_serving_evidence(serving_run_dir)
        runs.append(
            (
                "serving-recovery",
                serving,
                serving_resource(serving, anchor_unix_nano=anchor),
            )
        )
    if worker_run_dir is not None:
        worker = load_verified_worker_evidence(worker_run_dir)
        runs.append(
            (
                "worker-recovery",
                worker,
                worker_resource(worker, anchor_unix_nano=anchor),
            )
        )
    if not runs:
        raise TelemetryError("an export needs at least one source run directory")
    request = encode_export_request([resource for _, _, resource in runs])
    summary = validate_export_request(request)
    data = artifact_bytes(request)
    if len(data) > MAX_EXPORT_BYTES:
        raise TelemetryError(f"export exceeds its {MAX_EXPORT_BYTES}-byte bound")
    sources = tuple(_source(lab, resource) for lab, _, resource in runs)
    evidence = tuple(item for _, item, _ in runs)
    return TraceExport(request, data, sources, summary, evidence)


def build_trace_export_report(
    config: TelemetryConfig, export: TraceExport
) -> dict[str, Any]:
    """Seal the export digest, its sources, encoding and evidence boundary."""

    if not isinstance(config, TelemetryConfig):
        raise TypeError("config must be TelemetryConfig")
    if not isinstance(export, TraceExport):
        raise TypeError("export must be TraceExport")
    return seal_report(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "telemetry_scope": LOCAL_CPU_LAB,
            "evidence_boundary": TELEMETRY_BOUNDARY,
            "config": config.to_mapping(),
            "config_sha256": config.fingerprint(),
            "encoding": dict(ENCODING),
            "sources": [dict(source) for source in export.sources],
            "export": {
                "file": EXPORT_FILE,
                "bytes": len(export.data),
                "sha256": sha256_hex(export.data),
                **export.summary.to_mapping(),
            },
        }
    )


def write_trace_export(
    output_dir: Path,
    export: TraceExport,
    report: dict[str, Any],
    *,
    sources: Sequence[Path] = (),
) -> None:
    """Publish the export and then its sealed report into a new directory."""

    if not isinstance(export, TraceExport):
        raise TypeError("export must be TraceExport")
    require_sealed_report(report, kind=REPORT_KIND)
    if report["export"]["sha256"] != sha256_hex(export.data):
        raise TelemetryError("report does not seal these export bytes")
    create_output_directory(output_dir, sources=sources)
    publish_new_file(output_dir / EXPORT_FILE, export.data, max_bytes=MAX_EXPORT_BYTES)
    publish_new_file(
        output_dir / REPORT_FILE, artifact_bytes(report), max_bytes=MAX_REPORT_BYTES
    )


def load_trace_export_report(path: Path) -> dict[str, Any]:
    """Load a bounded canonical trace-export report and check its seal."""

    report = load_canonical_artifact(
        path, max_bytes=MAX_REPORT_BYTES, label="trace export report"
    )
    return require_sealed_report(report, kind=REPORT_KIND)
