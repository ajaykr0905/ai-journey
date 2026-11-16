import contextlib
import io
import json
from pathlib import Path

import pytest

from ai_journey import recovery_cli, serving_recovery_cli
from ai_journey.recovery_config import config_from_mapping as worker_config_from_mapping
from ai_journey.serving_recovery_config import load_serving_recovery_config
from ai_journey.telemetry_metrics_cli import main
from ai_journey.telemetry_metrics_config import load_telemetry_metrics_config
from ai_journey.telemetry_metrics_rules import build_rule_file, render_rule_file

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config" / "day-73-telemetry-metrics.json"
SERVING = ROOT / "config" / "day-70-serving-recovery.json"
WORKER = ROOT / "config" / "day-70-worker-recovery.json"
SOURCE = ROOT / "data" / "day-19-demo-names.txt"


def quiet(entry, argv):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        code = entry(argv)
    return code, output.getvalue()


@pytest.fixture(scope="module")
def pinned_evidence(tmp_path_factory):
    """Run the pinned Day 70 serving and worker experiments through their CLIs."""

    base = tmp_path_factory.mktemp("telemetry-pinned")
    serving, worker = base / "serving", base / "worker"
    code, _ = quiet(
        serving_recovery_cli.main,
        ["run", "--source", str(SOURCE), "--config", str(SERVING)]
        + ["--run-dir", str(serving)],
    )
    assert code == 0
    code, _ = quiet(
        recovery_cli.main,
        ["run", "--source", str(SOURCE), "--config", str(WORKER)]
        + ["--run-dir", str(worker)],
    )
    assert code == 0
    return base, serving, worker


def test_pinned_controls_parse_to_the_checked_in_mapping():
    config = load_telemetry_metrics_config(PINNED)

    assert config.to_mapping() == json.loads(PINNED.read_text(encoding="utf-8"))
    assert config.experiment_name == "day-73-telemetry-metrics"
    assert config.latency_buckets_us[0] == 250
    assert config.latency_buckets_us[-1] == 1_000_000
    assert len(config.latency_buckets_us) == 12


def test_pinned_objectives_follow_the_pinned_recovery_controls():
    config = load_telemetry_metrics_config(PINNED)
    serving = load_serving_recovery_config(SERVING)
    worker = worker_config_from_mapping(json.loads(WORKER.read_text()))

    assert config.serving.post_fault_latency_us == serving.slo_latency_ms * 1000
    assert config.serving.detection_latency_ms < serving.detection_timeout_ms
    assert config.serving.restart_latency_ms < serving.readiness_timeout_ms
    assert config.worker.detection_ms > worker.heartbeat_timeout_ms
    assert config.worker.recovery_ms == worker.worker_timeout_ms
    text = render_rule_file(build_rule_file(config))
    assert 'le="0.25",phase="post_fault"' in text
    assert "ai_journey_serving_recovery_detection_latency_seconds > 0.25'" in text


def test_pinned_export_of_the_pinned_runs_verifies(pinned_evidence):
    base, serving, worker = pinned_evidence
    output = base / "telemetry"
    evidence = ["--serving-run-dir", str(serving), "--worker-run-dir", str(worker)]

    code, printed = quiet(
        main,
        ["export", "--config", str(PINNED), "--output-dir", str(output), *evidence],
    )
    receipt = json.loads(printed)
    # Timings depend on this host's load, so a breach is reported, never hidden.
    assert code in (0, 3) and (code == 3) == bool(receipt["breached_objectives"])
    assert receipt["verified"] is True
    assert receipt["families"] == 28
    assert (receipt["recording_rules"], receipt["alerting_rules"]) == (3, 11)
    assert quiet(main, ["verify", "--output-dir", str(output), *evidence]) == (
        code,
        printed,
    )
    report = json.loads((output / "report.json").read_bytes())
    assert report["evidence"]["serving"]["requests"] == 48
    assert report["evidence"]["serving"]["run_id"].startswith(
        "day-70-serving-recovery-"
    )
    assert report["evidence"]["worker"]["run_id"].startswith("day-70-worker-recovery-")
