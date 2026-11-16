"""Independently verify a telemetry-metrics export from its evidence.

The verifier trusts nothing the export states. It loads the sealed report and
checks its schema, kind, scope and boundary, requires the copied config to be
canonical and equal to the report's, re-verifies both evidence directories
with their own verifiers, rederives the metric set, re-encodes the exposition
and regenerates the rules. The stored ``metrics.prom`` and ``rules.yml`` must
be exactly those bytes, and each is also parsed strictly on its own: the
exposition back into the same metric set and the rule file through the YAML
subset reader, the structural checks and a binding to the parsed exposition.
Key samples of the stored exposition are cross-checked against the verified
evidence reports directly, not through the adapters, so a mapping mistake in
an adapter cannot verify itself. Finally the whole report body, including
the objective judgements and the firing alerts, is recomputed and compared
section by section in canonical JSON. Export is deterministic, so the same
evidence under other objectives is an equally consistent export; given the
declared config, the verifier also requires the export to use exactly it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .serving_recovery_client import ATTEMPT_OUTCOMES
from .serving_recovery_config import (
    canonical_json_bytes,
    read_bounded_file,
    strict_json_loads,
)
from .serving_recovery_reconcile import REQUEST_OUTCOMES
from .telemetry_metrics_config import (
    MAX_CONFIG_BYTES,
    TelemetryMetricsConfig,
    config_from_mapping,
)
from .telemetry_metrics_exposition import (
    MAX_EXPOSITION_BYTES,
    encode_exposition,
    flatten_samples,
    parse_exposition,
)
from .telemetry_metrics_model import LAB_SCOPE, MetricSet, TelemetryMetricsError
from .telemetry_metrics_report import (
    CONFIG_FILE,
    EVIDENCE_BOUNDARY,
    EXPOSITION_FILE,
    REPORT_FILE,
    REPORT_KIND,
    REPORT_SCHEMA_VERSION,
    RULES_FILE,
    derive_metrics,
    export_receipt,
    load_metrics_report,
    report_body,
)
from .telemetry_metrics_rules import (
    build_rule_file,
    parse_rule_file,
    validate_rule_file,
)
from .telemetry_metrics_serving import ServingEvidence, load_verified_serving_evidence
from .telemetry_metrics_worker import WorkerEvidence, load_verified_worker_evidence
from .telemetry_metrics_yaml import MAX_YAML_BYTES

_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "scope",
        "config",
        "config_sha256",
        "evidence",
        "exposition",
        "rules",
        "objectives",
        "firing_alerts",
        "evidence_boundary",
        "report_sha256",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TelemetryMetricsError(message)


def _same(stored: object, recomputed: object) -> bool:
    """Compare JSON values by canonical bytes, so true, 1 and 1.0 differ."""

    return canonical_json_bytes(stored) == canonical_json_bytes(recomputed)


def _text(path: Path, max_bytes: int, name: str) -> tuple[bytes, str]:
    try:
        data = read_bounded_file(path, max_bytes=max_bytes, name=name)
        return data, data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TelemetryMetricsError(f"{name} is not UTF-8") from error
    except ValueError as error:
        raise TelemetryMetricsError(str(error)) from error


class _Samples:
    """Exact lookups of stored samples by name and labels."""

    def __init__(self, metrics: MetricSet) -> None:
        self._values: dict[tuple[str, frozenset], float] = {}
        for sample in flatten_samples(metrics):
            self._values[(sample.name, frozenset(sample.labels))] = sample.value

    def value(self, name: str, **labels: str) -> float | None:
        key = (name, frozenset({("scope", LAB_SCOPE), *labels.items()}))
        return self._values.get(key)

    def require(self, expected: float | None, name: str, **labels: str) -> None:
        """Require one sample, or its absence when ``expected`` is None."""

        stored = self.value(name, **labels)
        _require(
            stored == expected,
            f"stored {name}{labels or ''} is {stored}, the verified evidence says "
            f"{expected}",
        )


def _cross_check(
    metrics: MetricSet, serving: ServingEvidence, worker: WorkerEvidence
) -> None:
    """Re-read key samples against the verified reports, not the adapters."""

    samples = _Samples(metrics)
    report = serving.report
    reconciliation, timing = report["reconciliation"], report["timing"]
    totals = reconciliation["totals"]
    prefix = "ai_journey_serving_recovery_"
    for outcome in ATTEMPT_OUTCOMES:
        samples.require(
            reconciliation["attempt_outcomes"][outcome],
            prefix + "attempts_total",
            outcome=outcome,
        )
    for outcome in REQUEST_OUTCOMES:
        samples.require(totals[outcome], prefix + "requests_total", outcome=outcome)
    for name in ("duplicate_executions", "lost_response_executions"):
        samples.require(totals[name], prefix + name + "_total")
    for name in ("detection_latency", "restart_latency"):
        samples.require(timing[name + "_ns"] / 10**9, prefix + name + "_seconds")
    for name in ("unavailability", "time_to_slo_recovery"):
        nanoseconds = timing[name + "_ns"]
        samples.require(
            None if nanoseconds is None else nanoseconds / 10**9,
            prefix + name + "_seconds",
        )
    samples.require(int(report["recovered"]), prefix + "recovered")
    latency = prefix + "success_latency_seconds_count"
    samples.require(timing["pre_fault_latency"]["samples"], latency, phase="pre_fault")
    samples.require(
        timing["post_fault_latency"]["samples"], latency, phase="post_fault"
    )
    samples.require(
        timing["late_acknowledgements"], latency, phase="late_acknowledgement"
    )
    samples.require(
        1,
        prefix + "run_info",
        report_sha256=report["report_sha256"],
        run_id=report["experiment"]["run_id"],
    )
    worker_report = worker.report
    loss, worker_timing = worker_report["loss_assessment"], worker_report["timing"]
    prefix = "ai_journey_worker_recovery_"
    for name, field in (
        ("completed_items", "final_items"),
        ("lost_items", "lost_uncheckpointed_items"),
        ("replayed_items", "replayed_items"),
        ("omitted_items", "omitted_items"),
        ("duplicated_items", "duplicated_committed_items"),
    ):
        samples.require(loss[field], prefix + name + "_total")
    for name, start, end in (
        ("detection", "fault_injected_ns", "failure_detected_ns"),
        ("recovery", "fault_injected_ns", "recovery_completed_ns"),
    ):
        samples.require(
            (worker_timing[end] - worker_timing[start]) / 10**9,
            prefix + name + "_seconds",
        )
    samples.require(
        1,
        prefix + "run_info",
        report_sha256=worker_report["report_sha256"],
        run_id=worker_report["experiment"]["run_id"],
        trace_sha256=worker_report["trace_sha256"],
    )


def verify_metrics_export(
    output_dir: Path,
    *,
    serving_run_dir: Path,
    worker_run_dir: Path,
    declared_config: TelemetryMetricsConfig | None = None,
) -> dict[str, Any]:
    """Recompute an export from its evidence and require every byte to match.

    With ``declared_config`` the export's copied config must be exactly that
    config, so an export rebuilt under relaxed objectives cannot pass as one
    that met the declared ones. Returns the export receipt with
    ``verified: true``; raises ``TelemetryMetricsError`` on the first
    disagreement.
    """

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be pathlib.Path")
    if declared_config is not None and not isinstance(
        declared_config, TelemetryMetricsConfig
    ):
        raise TypeError("declared_config must be TelemetryMetricsConfig")
    _require(
        output_dir.is_dir() and not output_dir.is_symlink(), "output_dir is invalid"
    )
    report = load_metrics_report(output_dir / REPORT_FILE)
    _require(set(report) == _REPORT_FIELDS, "report fields do not match schema")
    _require(
        type(report["schema_version"]) is int
        and report["schema_version"] == REPORT_SCHEMA_VERSION
        and report["kind"] == REPORT_KIND,
        "report schema is unsupported",
    )
    _require(report["scope"] == LAB_SCOPE, "report scope is not local_cpu_lab")
    _require(report["evidence_boundary"] == EVIDENCE_BOUNDARY, "boundary changed")

    config_bytes, _ = _text(output_dir / CONFIG_FILE, MAX_CONFIG_BYTES, "config")
    try:
        config = config_from_mapping(strict_json_loads(config_bytes, name="config"))
    except TelemetryMetricsError:
        raise
    except ValueError as error:
        raise TelemetryMetricsError(str(error)) from error
    _require(
        config_bytes == canonical_json_bytes(config.to_mapping()),
        "config copy is not canonical JSON",
    )
    _require(
        _same(report["config"], config.to_mapping())
        and report["config_sha256"] == config.fingerprint(),
        "report config does not match the copied config",
    )
    _require(
        declared_config is None
        or config.fingerprint() == declared_config.fingerprint(),
        "export config is not the declared config",
    )

    serving = load_verified_serving_evidence(serving_run_dir)
    worker = load_verified_worker_evidence(worker_run_dir)
    metrics = derive_metrics(config, serving, worker)
    rule_file = build_rule_file(config)

    exposition_bytes, exposition = _text(
        output_dir / EXPOSITION_FILE, MAX_EXPOSITION_BYTES, "exposition"
    )
    stored = parse_exposition(exposition)
    _require(stored == metrics, "stored exposition samples differ from the evidence")
    _require(
        exposition_bytes == encode_exposition(metrics).encode("utf-8"),
        "stored exposition is not the canonical encoding",
    )
    _cross_check(stored, serving, worker)

    _, rules = _text(output_dir / RULES_FILE, MAX_YAML_BYTES, "rules")
    stored_rules = parse_rule_file(rules)
    _require(
        stored_rules == rule_file, "stored rules differ from the declared objectives"
    )
    validate_rule_file(stored_rules, stored)

    body = report_body(config, serving, worker, metrics, rule_file)
    for section, value in body.items():
        _require(
            _same(report[section], value),
            f"report {section} does not match the recomputed export",
        )
    return {"verified": True, **export_receipt(report)}
