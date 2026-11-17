import json
from hashlib import sha256

import pytest
from test_telemetry_metrics_config import valid_config
from test_telemetry_metrics_rules import RELAXED
from test_telemetry_metrics_serving import make_serving_run
from test_telemetry_metrics_worker import make_worker_run

import ai_journey.telemetry_metrics_report as report_module
from ai_journey.serving_recovery_config import canonical_json_bytes
from ai_journey.telemetry_metrics_exposition import parse_exposition
from ai_journey.telemetry_metrics_model import TelemetryMetricsError
from ai_journey.telemetry_metrics_report import (
    EVIDENCE_BOUNDARY,
    build_metrics_export,
    export_receipt,
    load_metrics_report,
    require_sealed_metrics_report,
    seal_metrics_report,
    write_metrics_export,
)
from ai_journey.telemetry_metrics_rules import parse_rule_file

RELAXED_CONFIG = valid_config(**RELAXED)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("telemetry-report")
    return make_serving_run(base), make_worker_run(base)


def export(runs, config=RELAXED_CONFIG):
    serving, worker = runs
    return build_metrics_export(config, serving_run_dir=serving, worker_run_dir=worker)


def test_report_binds_config_evidence_exposition_rules_and_boundary(runs):
    built = export(runs)
    report = built.report
    serving_report = json.loads((runs[0] / "report.json").read_bytes())
    worker_report = json.loads((runs[1] / "report.json").read_bytes())

    assert report["kind"] == "ai-journey-telemetry-metrics-report"
    assert report["scope"] == "local_cpu_lab"
    assert report["config"] == RELAXED_CONFIG.to_mapping()
    assert report["config_sha256"] == RELAXED_CONFIG.fingerprint()
    assert (
        report["evidence"]["serving"]["report_sha256"]
        == (serving_report["report_sha256"])
    )
    assert report["evidence"]["serving"]["recovered"] is True
    assert report["evidence"]["worker"] == {
        "run_id": worker_report["experiment"]["run_id"],
        "report_sha256": worker_report["report_sha256"],
        "trace_sha256": worker_report["trace_sha256"],
    }
    exposition, rules = report["exposition"], report["rules"]
    assert exposition["file_sha256"] == sha256(built.exposition.encode()).hexdigest()
    assert exposition["bytes"] == len(built.exposition.encode())
    assert exposition["content_type"] == "text/plain; version=0.0.4; charset=utf-8"
    assert parse_exposition(built.exposition) == built.metrics
    assert exposition["families"] == len(built.metrics.families) == 16 + 12
    assert exposition["samples"] == len(
        [line for line in built.exposition.splitlines() if not line.startswith("#")]
    )
    assert rules["file_sha256"] == sha256(built.rules.encode()).hexdigest()
    assert (rules["groups"], rules["recording_rules"], rules["alerting_rules"]) == (
        2,
        3,
        11,
    )
    assert parse_rule_file(built.rules).groups[0].name.endswith("serving_recovery_slo")
    assert [item["slo"] for item in report["objectives"]][:2] == [
        "serving-detection-latency",
        "serving-restart-latency",
    ]
    assert not any(item["breached"] for item in report["objectives"])
    assert report["firing_alerts"] == []
    assert report["evidence_boundary"] == EVIDENCE_BOUNDARY
    for phrase in ("Local CPU lab telemetry", "not GPU", "production", "one host"):
        assert phrase in EVIDENCE_BOUNDARY
    assert require_sealed_metrics_report(report) is report
    assert export(runs).report == report


def test_breached_objectives_and_firing_alerts_must_agree(runs, monkeypatch):
    tight = RELAXED["serving_objectives"] | {"restart_latency_ms": 1}
    report = export(runs, valid_config(**(RELAXED | {"serving_objectives": tight})))
    breached = [item for item in report.report["objectives"] if item["breached"]]

    assert [item["slo"] for item in breached] == ["serving-restart-latency"]
    assert breached[0]["statement"] == "restart_latency_ns <= 1000000"
    assert breached[0]["observed"] > 1_000_000
    (alert,) = report.report["firing_alerts"]
    assert alert["alert"] == breached[0]["alert"] == "ServingRecoveryRestartSlow"
    assert alert["labels"]["slo"] == "serving-restart-latency"
    assert float(alert["value"]) == breached[0]["observed"] / 1e9
    receipt = export_receipt(report.report)
    assert receipt["breached_objectives"] == ["serving-restart-latency"]
    assert receipt["firing_alerts"] == ["ServingRecoveryRestartSlow"]
    assert (receipt["families"], receipt["alerting_rules"]) == (28, 11)

    monkeypatch.setattr(report_module, "evaluate_rule_file", lambda *_: ())
    with pytest.raises(TelemetryMetricsError, match="disagree"):
        export(runs, valid_config(**(RELAXED | {"serving_objectives": tight})))


def test_latency_objective_counts_post_fault_attempts_exactly(runs):
    strict = RELAXED["serving_objectives"] | {
        "post_fault_latency_us": 250,
        "post_fault_within_latency_ratio": 1.0,
    }
    report = export(runs, valid_config(**(RELAXED | {"serving_objectives": strict})))
    (latency,) = [
        item
        for item in report.report["objectives"]
        if item["slo"] == "serving-post-fault-latency"
    ]

    assert latency["statement"] == (
        "post-fault successes within 250 us / post-fault successes >= 1.0"
    )
    assert 0 <= latency["observed"] <= 1
    fired = {alert["alert"] for alert in report.report["firing_alerts"]}
    assert ("ServingRecoveryPostFaultLatencyMissed" in fired) is latency["breached"]


def test_exports_publish_once_into_a_new_directory_outside_evidence(runs, tmp_path):
    built = export(runs)
    output = tmp_path / "export"

    write_metrics_export(output, built, evidence_dirs=runs)
    assert sorted(path.name for path in output.iterdir()) == [
        "config.json",
        "metrics.prom",
        "report.json",
        "rules.yml",
    ]
    assert (output / "metrics.prom").read_text() == built.exposition
    assert (output / "rules.yml").read_text() == built.rules
    assert json.loads((output / "config.json").read_bytes()) == (
        RELAXED_CONFIG.to_mapping()
    )
    assert load_metrics_report(output / "report.json") == built.report
    with pytest.raises(TelemetryMetricsError, match="must not already exist"):
        write_metrics_export(output, built)
    with pytest.raises(TelemetryMetricsError, match="inside evidence"):
        write_metrics_export(runs[0] / "export", built, evidence_dirs=runs)
    assert not (runs[0] / "export").exists()
    with pytest.raises(TypeError):
        write_metrics_export(str(tmp_path / "other"), built)


def test_reports_reject_tampering_and_contain_no_host_paths(runs, tmp_path):
    built = export(runs)
    encoded = canonical_json_bytes(built.report)
    for marker in (
        b"/Users/",
        b"/home/",
        b"/private/",
        b"/tmp/",
        str(runs[0]).encode(),
    ):
        assert marker not in encoded

    tampered = dict(built.report, scope="production")
    (tmp_path / "tampered.json").write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(TelemetryMetricsError, match="seal"):
        load_metrics_report(tmp_path / "tampered.json")
    (tmp_path / "spaced.json").write_text(json.dumps(built.report, indent=1))
    with pytest.raises(TelemetryMetricsError, match="canonical"):
        load_metrics_report(tmp_path / "spaced.json")
    with pytest.raises(TelemetryMetricsError, match="readable regular"):
        load_metrics_report(tmp_path / "missing.json")
    with pytest.raises(TelemetryMetricsError, match="unsealed"):
        seal_metrics_report(built.report)
    with pytest.raises(TelemetryMetricsError, match="not sealed"):
        require_sealed_metrics_report([])


def test_exports_need_a_config_and_verifiable_evidence(runs, tmp_path):
    with pytest.raises(TypeError):
        build_metrics_export(
            RELAXED_CONFIG.to_mapping(), serving_run_dir=runs[0], worker_run_dir=runs[1]
        )
    with pytest.raises(TelemetryMetricsError, match="serving-recovery evidence"):
        build_metrics_export(
            RELAXED_CONFIG, serving_run_dir=runs[1], worker_run_dir=runs[1]
        )
    with pytest.raises(TelemetryMetricsError, match="worker-recovery evidence"):
        build_metrics_export(
            RELAXED_CONFIG, serving_run_dir=runs[0], worker_run_dir=runs[0]
        )
