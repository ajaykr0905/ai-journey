"""Sealed dashboard reports and their independent verifier.

``build_dashboard_artifact`` renders the dashboard from a pinned config,
validates it, requires every catalog metric to be charted and renders
canonical bytes. The sealed report binds those bytes by SHA-256 to the config,
the metric catalog, the Grafana schema version, the local CPU lab scope and
evidence boundary, and states that the dashboard was not imported into a
running Grafana. ``verify_dashboard_report`` trusts none of it: it checks the
seal and schema, optionally binds the embedded config to a pinned file,
re-reads and re-validates the dashboard file and rebuilds it from the
embedded config to require the same bytes.

A report that cannot be read raises ``TelemetryError``; anything read and
then found wrong raises ``TelemetryVerificationError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .telemetry_dashboard_grafana import (
    DASHBOARD_BOUNDARY,
    DASHBOARD_SCHEMA_VERSION,
    DATASOURCE_VARIABLE,
    build_dashboard,
)
from .telemetry_dashboard_metrics import METRICS
from .telemetry_dashboard_validate import DashboardSummary, validate_dashboard
from .telemetry_traces_config import (
    TelemetryConfig,
    config_from_mapping,
    load_telemetry_config,
)
from .telemetry_traces_io import (
    TelemetryError,
    TelemetryVerificationError,
    artifact_bytes,
    create_output_directory,
    publish_new_file,
    read_bounded_file,
    require_sealed_report,
    seal_report,
    sha256_hex,
    strict_json_loads,
)
from .telemetry_traces_validate import LOCAL_CPU_LAB

DASHBOARD_FILE = "dashboard.json"
REPORT_FILE = "report.json"
REPORT_KIND = "ai-journey-telemetry-dashboard"
REPORT_SCHEMA_VERSION = 1
MAX_DASHBOARD_BYTES = 1024 * 1024
MAX_REPORT_BYTES = 256 * 1024
_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "telemetry_scope",
        "evidence_boundary",
        "config",
        "config_sha256",
        "grafana",
        "metrics",
        "dashboard",
        "report_sha256",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TelemetryVerificationError(message)


def _grafana_section(config: TelemetryConfig) -> dict[str, Any]:
    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "uid": config.dashboard_uid,
        "title": config.dashboard_title,
        "datasource": f"prometheus through the ${{{DATASOURCE_VARIABLE}}} variable",
        "imported_into_grafana": False,
    }


def uncharted_metrics(summary: DashboardSummary) -> list[str]:
    """Return catalog metrics that no dashboard query selects."""

    charted = set(summary.series)
    return [m.name for m in METRICS if not charted & set(m.series)]


@dataclass(frozen=True)
class DashboardArtifact:
    """A validated dashboard, its canonical bytes and its query summary."""

    dashboard: dict[str, Any]
    data: bytes
    summary: DashboardSummary


def build_dashboard_artifact(config: TelemetryConfig) -> DashboardArtifact:
    """Render, validate and encode the dashboard of one pinned config."""

    dashboard = build_dashboard(config)
    summary = validate_dashboard(dashboard)
    missing = uncharted_metrics(summary)
    if missing:
        raise TelemetryError(f"dashboard does not chart {', '.join(missing)}")
    data = artifact_bytes(dashboard)
    if len(data) > MAX_DASHBOARD_BYTES:
        raise TelemetryError(f"dashboard exceeds its {MAX_DASHBOARD_BYTES}-byte bound")
    return DashboardArtifact(dashboard, data, summary)


def build_dashboard_report(
    config: TelemetryConfig, artifact: DashboardArtifact
) -> dict[str, Any]:
    """Seal the dashboard digest with its config, catalog and boundary."""

    if not isinstance(config, TelemetryConfig):
        raise TypeError("config must be TelemetryConfig")
    if not isinstance(artifact, DashboardArtifact):
        raise TypeError("artifact must be DashboardArtifact")
    return seal_report(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "telemetry_scope": LOCAL_CPU_LAB,
            "evidence_boundary": DASHBOARD_BOUNDARY,
            "config": config.to_mapping(),
            "config_sha256": config.fingerprint(),
            "grafana": _grafana_section(config),
            "metrics": [metric.to_mapping() for metric in METRICS],
            "dashboard": {
                "file": DASHBOARD_FILE,
                "bytes": len(artifact.data),
                "sha256": sha256_hex(artifact.data),
                **artifact.summary.to_mapping(),
            },
        }
    )


def write_dashboard_artifact(
    output_dir: Path, artifact: DashboardArtifact, report: dict[str, Any]
) -> None:
    """Publish the dashboard and then its sealed report into a new directory."""

    if not isinstance(artifact, DashboardArtifact):
        raise TypeError("artifact must be DashboardArtifact")
    require_sealed_report(report, kind=REPORT_KIND)
    if report["dashboard"]["sha256"] != sha256_hex(artifact.data):
        raise TelemetryError("report does not seal these dashboard bytes")
    create_output_directory(output_dir)
    publish_new_file(
        output_dir / DASHBOARD_FILE, artifact.data, max_bytes=MAX_DASHBOARD_BYTES
    )
    publish_new_file(
        output_dir / REPORT_FILE, artifact_bytes(report), max_bytes=MAX_REPORT_BYTES
    )


def verify_dashboard_report(
    report_path: Path, *, config_path: Path | None = None
) -> dict[str, Any]:
    """Re-validate and rebuild a sealed dashboard and return a receipt."""

    data = read_bounded_file(
        report_path, max_bytes=MAX_REPORT_BYTES, label="dashboard report"
    )
    pinned = load_telemetry_config(config_path) if config_path is not None else None
    try:
        report = strict_json_loads(data, label="dashboard report")
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
    _require(report["evidence_boundary"] == DASHBOARD_BOUNDARY, "boundary changed")
    _require(report["config_sha256"] == config.fingerprint(), "config digest differs")
    _require(
        pinned is None or pinned == config,
        "report was built from another config than the pinned one",
    )
    _require(report["grafana"] == _grafana_section(config), "grafana section changed")
    _require(
        report["metrics"] == [metric.to_mapping() for metric in METRICS],
        "metric catalog differs from the one this lab exposes",
    )
    stored = report["dashboard"]
    _require(
        type(stored) is dict and stored.get("file") == DASHBOARD_FILE,
        "report names an unexpected dashboard file",
    )
    try:
        dashboard_bytes = read_bounded_file(
            report_path.parent / DASHBOARD_FILE,
            max_bytes=MAX_DASHBOARD_BYTES,
            label="dashboard",
        )
        dashboard = strict_json_loads(dashboard_bytes, label="dashboard")
    except TelemetryError as error:
        raise TelemetryVerificationError(str(error)) from error
    _require(
        stored.get("bytes") == len(dashboard_bytes)
        and stored.get("sha256") == sha256_hex(dashboard_bytes),
        "dashboard bytes do not match the sealed digest",
    )
    _require(
        artifact_bytes(dashboard) == dashboard_bytes, "dashboard is not canonical JSON"
    )
    summary = validate_dashboard(dashboard)
    _require(
        {key: stored[key] for key in stored if key not in ("file", "bytes", "sha256")}
        == summary.to_mapping(),
        "dashboard summary does not match its contents",
    )
    missing = uncharted_metrics(summary)
    _require(not missing, f"dashboard does not chart {', '.join(missing)}")
    _require(
        build_dashboard_artifact(config).data == dashboard_bytes,
        "dashboard does not rebuild from the embedded config",
    )
    return {
        "verified": True,
        "kind": REPORT_KIND,
        "telemetry_scope": LOCAL_CPU_LAB,
        "report_sha256": report["report_sha256"],
        "dashboard_sha256": stored["sha256"],
        "uid": summary.uid,
        "schema_version": summary.schema_version,
        "panels": summary.panels,
        "targets": summary.targets,
    }
