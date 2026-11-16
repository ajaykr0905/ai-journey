"""Sealed telemetry-metrics export bound to verified recovery evidence.

``build_metrics_export`` loads a serving run directory and a worker run
directory through the verifying adapters, derives one metric set, encodes the
exposition, generates the rules from the config's objectives, binds them to
that exposition and evaluates them at one instant. Each declared objective is
also judged directly from the verified evidence in exact integer nanoseconds
and counts, and the export is refused unless both judgements breach the same
objectives. The report binds the config, both evidence seals and the
exposition and rule bytes by SHA-256, and records the objective results, the
firing alerts and the evidence boundary. Its seal is the SHA-256 of the
canonical report without the seal field: it proves content identity, not
authorship, which is why ``telemetry_metrics_verify`` recomputes everything.

``write_metrics_export`` publishes ``config.json``, ``metrics.prom``,
``rules.yml`` and, last, ``report.json`` into a new directory with the serving
lab's no-overwrite atomic writer, so a reader that finds the report finds a
complete export.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .serving_recovery_config import canonical_json_bytes, canonical_sha256, sha256_hex
from .serving_recovery_identity import lifetime_number
from .serving_recovery_io import load_canonical_json, publish_new_file
from .telemetry_metrics_config import TelemetryMetricsConfig
from .telemetry_metrics_exposition import (
    CONTENT_TYPE,
    MAX_EXPOSITION_BYTES,
    encode_exposition,
    flatten_samples,
)
from .telemetry_metrics_model import (
    LAB_SCOPE,
    MetricSet,
    TelemetryMetricsError,
    metric_set,
)
from .telemetry_metrics_rules import (
    AlertingRule,
    FiringAlert,
    RecordingRule,
    RuleFile,
    build_rule_file,
    declared_objectives,
    evaluate_rule_file,
    render_rule_file,
)
from .telemetry_metrics_serving import (
    ServingEvidence,
    load_verified_serving_evidence,
    serving_recovery_families,
)
from .telemetry_metrics_worker import (
    WorkerEvidence,
    load_verified_worker_evidence,
    worker_recovery_families,
)
from .telemetry_metrics_yaml import MAX_YAML_BYTES

CONFIG_FILE = "config.json"
EXPOSITION_FILE = "metrics.prom"
RULES_FILE = "rules.yml"
REPORT_FILE = "report.json"
REPORT_KIND = "ai-journey-telemetry-metrics-report"
REPORT_SCHEMA_VERSION = 1
MAX_REPORT_BYTES = 256 * 1024
EVIDENCE_BOUNDARY = (
    "Local CPU lab telemetry: every sample is derived offline from the sealed "
    "evidence of one serving-recovery run and one worker-recovery run on one "
    "host, after their own verifiers accepted it. No Prometheus server scraped "
    "a live target, and no Grafana dashboard, OpenTelemetry collector or "
    "Alertmanager ran; the rules were evaluated locally on one snapshot. This "
    "is not GPU, CUDA, distributed, multi-host or production monitoring "
    "evidence, and the objectives are declared lab controls, not service-level "
    "agreements. Durations come from time.monotonic_ns on the experiment host "
    "and depend on its load."
)


@dataclass(frozen=True)
class MetricsExport:
    """One export: its metric set, exposition and rule texts, and sealed report."""

    config: TelemetryMetricsConfig
    metrics: MetricSet
    exposition: str
    rules: str
    report: dict[str, Any]


def _duration(
    slo: str, field: str, limit_ms: int, observed: int | None
) -> dict[str, Any]:
    limit = limit_ms * 1_000_000
    return {
        "slo": slo,
        "statement": f"{field} <= {limit}",
        "observed": observed,
        "breached": observed is not None and observed > limit,
    }


def _zero(slo: str, field: str, observed: int) -> dict[str, Any]:
    return {
        "slo": slo,
        "statement": f"{field} == 0",
        "observed": observed,
        "breached": observed != 0,
    }


def judge_objectives(
    config: TelemetryMetricsConfig,
    serving: ServingEvidence,
    worker: WorkerEvidence,
) -> list[dict[str, Any]]:
    """Judge every declared objective straight from the verified evidence.

    This path never reads the exposition: durations stay integer nanoseconds
    and the latency ratio counts attempts, so it can check the rules.
    """

    report, timing = serving.report, serving.report["timing"]
    worker_timing = worker.report["timing"]
    loss = worker.report["loss_assessment"]
    objectives = config.serving
    faulted = lifetime_number(report["fault"]["lifetime_id"])
    post_fault = [
        attempt.latency_ns
        for attempt in serving.attempts
        if attempt.outcome == "ok" and lifetime_number(attempt.lifetime_id) > faulted
    ]
    within = sum(
        latency <= objectives.post_fault_latency_us * 1000 for latency in post_fault
    )
    ratio = within / len(post_fault) if post_fault else None
    target = objectives.post_fault_within_latency_ratio
    judges: dict[str, Callable[[], dict[str, Any]]] = {
        "serving-detection-latency": lambda: _duration(
            "serving-detection-latency",
            "detection_latency_ns",
            objectives.detection_latency_ms,
            timing["detection_latency_ns"],
        ),
        "serving-restart-latency": lambda: _duration(
            "serving-restart-latency",
            "restart_latency_ns",
            objectives.restart_latency_ms,
            timing["restart_latency_ns"],
        ),
        "serving-unavailability": lambda: _duration(
            "serving-unavailability",
            "unavailability_ns",
            objectives.unavailability_ms,
            timing["unavailability_ns"],
        ),
        "serving-time-to-slo-recovery": lambda: _duration(
            "serving-time-to-slo-recovery",
            "time_to_slo_recovery_ns",
            objectives.time_to_slo_recovery_ms,
            timing["time_to_slo_recovery_ns"],
        ),
        "serving-recovered": lambda: {
            "slo": "serving-recovered",
            "statement": "recovered == true",
            "observed": report["recovered"],
            "breached": report["recovered"] is not True,
        },
        "serving-request-success": lambda: _zero(
            "serving-request-success",
            "failed_after_budget",
            report["reconciliation"]["totals"]["failed_after_budget"],
        ),
        "serving-post-fault-latency": lambda: {
            "slo": "serving-post-fault-latency",
            "statement": (
                f"post-fault successes within {objectives.post_fault_latency_us} us "
                f"/ post-fault successes >= {target!r}"
            ),
            "observed": ratio,
            "breached": ratio is not None and ratio < target,
        },
        "worker-detection": lambda: _duration(
            "worker-detection",
            "detection_ns",
            config.worker.detection_ms,
            worker_timing["failure_detected_ns"] - worker_timing["fault_injected_ns"],
        ),
        "worker-recovery-time": lambda: _duration(
            "worker-recovery-time",
            "recovery_ns",
            config.worker.recovery_ms,
            worker_timing["recovery_completed_ns"] - worker_timing["fault_injected_ns"],
        ),
        "worker-no-omitted-items": lambda: _zero(
            "worker-no-omitted-items", "omitted_items", loss["omitted_items"]
        ),
        "worker-no-duplicated-items": lambda: _zero(
            "worker-no-duplicated-items",
            "duplicated_committed_items",
            loss["duplicated_committed_items"],
        ),
    }
    judged = []
    for _, group in declared_objectives(config):
        for objective in group:
            judge = judges.pop(objective.slo, None)
            if judge is None:
                raise TelemetryMetricsError(f"no evidence judge for {objective.slo}")
            judged.append({"alert": objective.alert, **judge()})
    if judges:
        raise TelemetryMetricsError(f"judges without objectives: {sorted(judges)}")
    return judged


def alert_mapping(alert: FiringAlert) -> dict[str, Any]:
    return {"alert": alert.alert, "labels": dict(alert.labels), "value": alert.value}


def derive_metrics(
    config: TelemetryMetricsConfig, serving: ServingEvidence, worker: WorkerEvidence
) -> MetricSet:
    """The one metric set both adapters derive from verified evidence."""

    return metric_set(
        serving_recovery_families(serving, latency_buckets_us=config.latency_buckets_us)
        + worker_recovery_families(worker)
    )


def report_body(
    config: TelemetryMetricsConfig,
    serving: ServingEvidence,
    worker: WorkerEvidence,
    metrics: MetricSet,
    rule_file: RuleFile,
) -> dict[str, Any]:
    """Assemble the unsealed report; refuse disagreeing objective judgements."""

    exposition = encode_exposition(metrics).encode("utf-8")
    rules = render_rule_file(rule_file).encode("utf-8")
    firing = evaluate_rule_file(rule_file, metrics)
    objectives = judge_objectives(config, serving, worker)
    if sorted({alert.alert for alert in firing}) != sorted(
        item["alert"] for item in objectives if item["breached"]
    ):
        raise TelemetryMetricsError(
            "rule evaluation and the evidence disagree on breached objectives"
        )
    receipt = serving.receipt
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "scope": LAB_SCOPE,
        "config": config.to_mapping(),
        "config_sha256": config.fingerprint(),
        "evidence": {
            "serving": {
                "run_id": receipt["run_id"],
                "report_sha256": receipt["report_sha256"],
                "recovered": receipt["recovered"],
                "requests": receipt["requests"],
                "duplicate_executions": receipt["duplicate_executions"],
            },
            "worker": {
                "run_id": worker.report["experiment"]["run_id"],
                "report_sha256": worker.report["report_sha256"],
                "trace_sha256": worker.report["trace_sha256"],
            },
        },
        "exposition": {
            "file": EXPOSITION_FILE,
            "file_sha256": sha256_hex(exposition),
            "bytes": len(exposition),
            "content_type": CONTENT_TYPE,
            "families": len(metrics.families),
            "series": metrics.series_count,
            "samples": len(flatten_samples(metrics)),
        },
        "rules": {
            "file": RULES_FILE,
            "file_sha256": sha256_hex(rules),
            "bytes": len(rules),
            "groups": len(rule_file.groups),
            "recording_rules": sum(type(r) is RecordingRule for r in rule_file.rules),
            "alerting_rules": sum(type(r) is AlertingRule for r in rule_file.rules),
        },
        "objectives": objectives,
        "firing_alerts": [alert_mapping(alert) for alert in firing],
        "evidence_boundary": EVIDENCE_BOUNDARY,
    }


def seal_metrics_report(report: dict[str, Any]) -> dict[str, Any]:
    """Add the canonical seal over every other report field."""

    if type(report) is not dict or "report_sha256" in report:
        raise TelemetryMetricsError("report must be an unsealed mapping")
    return {**report, "report_sha256": canonical_sha256(report)}


def require_sealed_metrics_report(report: object) -> dict[str, Any]:
    """Return a report whose seal matches its canonical content."""

    if type(report) is not dict or type(report.get("report_sha256")) is not str:
        raise TelemetryMetricsError("metrics report is not sealed")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if canonical_sha256(unsigned) != report["report_sha256"]:
        raise TelemetryMetricsError("metrics report seal does not match its content")
    return report


def build_metrics_export(
    config: TelemetryMetricsConfig, *, serving_run_dir: Path, worker_run_dir: Path
) -> MetricsExport:
    """Verify both evidence directories, then derive and seal one export."""

    if not isinstance(config, TelemetryMetricsConfig):
        raise TypeError("config must be TelemetryMetricsConfig")
    serving = load_verified_serving_evidence(serving_run_dir)
    worker = load_verified_worker_evidence(worker_run_dir)
    metrics = derive_metrics(config, serving, worker)
    rule_file = build_rule_file(config)
    report = seal_metrics_report(
        report_body(config, serving, worker, metrics, rule_file)
    )
    return MetricsExport(
        config,
        metrics,
        encode_exposition(metrics),
        render_rule_file(rule_file),
        report,
    )


def _inside(path: Path, directory: Path) -> bool:
    return path.resolve().is_relative_to(directory.resolve())


def write_metrics_export(
    output_dir: Path,
    export: MetricsExport,
    *,
    evidence_dirs: tuple[Path, ...] = (),
) -> None:
    """Publish an export into a new directory that no evidence directory holds."""

    if not isinstance(output_dir, Path) or type(export) is not MetricsExport:
        raise TypeError("write_metrics_export needs a Path and a MetricsExport")
    require_sealed_metrics_report(export.report)
    if output_dir.exists() or output_dir.is_symlink():
        raise TelemetryMetricsError("output_dir must not already exist")
    if any(_inside(output_dir, directory) for directory in evidence_dirs):
        raise TelemetryMetricsError("output_dir must not be inside evidence")
    output_dir.mkdir(parents=True)
    files = (
        (CONFIG_FILE, canonical_json_bytes(export.config.to_mapping()), 16 * 1024),
        (EXPOSITION_FILE, export.exposition.encode("utf-8"), MAX_EXPOSITION_BYTES),
        (RULES_FILE, export.rules.encode("utf-8"), MAX_YAML_BYTES),
        (REPORT_FILE, canonical_json_bytes(export.report), MAX_REPORT_BYTES),
    )
    try:
        for name, data, bound in files:
            publish_new_file(output_dir / name, data, max_bytes=bound)
    except ValueError as error:
        raise TelemetryMetricsError(str(error)) from error


def load_metrics_report(path: Path) -> dict[str, Any]:
    """Load a bounded canonical report and check its seal."""

    try:
        report = load_canonical_json(path, max_bytes=MAX_REPORT_BYTES, name="report")
    except ValueError as error:
        raise TelemetryMetricsError(str(error)) from error
    return require_sealed_metrics_report(report)


def export_receipt(report: dict[str, Any]) -> dict[str, Any]:
    """The short summary that the CLI prints for one export."""

    exposition, rules = report["exposition"], report["rules"]
    return {
        "report_sha256": report["report_sha256"],
        "scope": report["scope"],
        "families": exposition["families"],
        "series": exposition["series"],
        "samples": exposition["samples"],
        "recording_rules": rules["recording_rules"],
        "alerting_rules": rules["alerting_rules"],
        "breached_objectives": [
            item["slo"] for item in report["objectives"] if item["breached"]
        ],
        "firing_alerts": sorted({alert["alert"] for alert in report["firing_alerts"]}),
    }
