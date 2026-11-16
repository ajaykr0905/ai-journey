import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_telemetry_metrics_config import config_mapping
from test_telemetry_metrics_rules import RELAXED
from test_telemetry_metrics_serving import make_serving_run
from test_telemetry_metrics_worker import make_worker_run, rewrite

from ai_journey.serving_recovery_config import canonical_json_bytes
from ai_journey.telemetry_metrics_cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    base = tmp_path_factory.mktemp("telemetry-cli")
    config = base / "telemetry-config.json"
    config.write_bytes(canonical_json_bytes(config_mapping(**RELAXED)))
    breached = base / "breached-config.json"
    tight = RELAXED["serving_objectives"] | {"restart_latency_ms": 1}
    breached.write_bytes(
        canonical_json_bytes(
            config_mapping(**(RELAXED | {"serving_objectives": tight}))
        )
    )
    return base, make_serving_run(base), make_worker_run(base), config, breached


def export_argv(evidence, output, config=None):
    _, serving, worker, relaxed, _ = evidence
    return [
        "export",
        "--config",
        str(config or relaxed),
        "--serving-run-dir",
        str(serving),
        "--worker-run-dir",
        str(worker),
        "--output-dir",
        str(output),
    ]


def verify_argv(evidence, output, serving=None):
    _, real_serving, worker, _, _ = evidence
    return [
        "verify",
        "--output-dir",
        str(output),
        "--serving-run-dir",
        str(serving or real_serving),
        "--worker-run-dir",
        str(worker),
    ]


def test_export_publishes_and_verify_recomputes_the_same_receipt(
    evidence, tmp_path, capsys
):
    output = tmp_path / "export"

    assert main(export_argv(evidence, output)) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["verified"] is True and exported["scope"] == "local_cpu_lab"
    assert exported["breached_objectives"] == exported["firing_alerts"] == []
    assert sorted(path.name for path in output.iterdir()) == [
        "config.json",
        "metrics.prom",
        "report.json",
        "rules.yml",
    ]
    assert main(verify_argv(evidence, output)) == 0
    assert json.loads(capsys.readouterr().out) == exported

    module = subprocess.run(
        [sys.executable, "-m", "ai_journey.telemetry_metrics_cli"]
        + verify_argv(evidence, output),
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=60,
    )
    assert module.returncode == 0, module.stderr
    assert json.loads(module.stdout) == exported


def test_breached_objectives_exit_three_after_full_verification(
    evidence, tmp_path, capsys
):
    output = tmp_path / "breached"

    assert main(export_argv(evidence, output, config=evidence[4])) == 3
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["verified"] is True
    assert receipt["breached_objectives"] == ["serving-restart-latency"]
    assert receipt["firing_alerts"] == ["ServingRecoveryRestartSlow"]
    assert main(verify_argv(evidence, output)) == 3
    assert json.loads(capsys.readouterr().out) == receipt


def test_invalid_inputs_exit_two_without_publishing(evidence, tmp_path, capsys):
    serving = evidence[1]
    output = tmp_path / "export"
    output.mkdir()

    assert main(export_argv(evidence, output)) == 2
    assert "must not already exist" in capsys.readouterr().err
    broken = tmp_path / "broken.json"
    broken.write_text('{"schema_version": 1, "schema_version": 1}')
    assert main(export_argv(evidence, tmp_path / "new", config=broken)) == 2
    assert "duplicate JSON key" in capsys.readouterr().err
    argv = export_argv(evidence, tmp_path / "new")
    argv[argv.index("--worker-run-dir") + 1] = str(serving)
    assert main(argv) == 2
    assert "worker-recovery evidence was rejected" in capsys.readouterr().err
    assert main(export_argv(evidence, serving / "telemetry")) == 2
    assert "inside evidence" in capsys.readouterr().err
    assert not (tmp_path / "new").exists() and not (serving / "telemetry").exists()


def test_rejected_exports_exit_one(evidence, tmp_path, capsys):
    output = tmp_path / "export"
    assert main(export_argv(evidence, output)) == 0
    capsys.readouterr()

    other = make_serving_run(tmp_path)
    assert main(verify_argv(evidence, output, serving=other)) == 1
    assert "verification failed" in capsys.readouterr().err
    exposition = output / "metrics.prom"
    exposition.write_text(exposition.read_text().replace("# HELP", "#  HELP", 1))
    assert main(verify_argv(evidence, output)) == 1
    assert "comment lines" in capsys.readouterr().err
    assert main(verify_argv(evidence, tmp_path / "missing")) == 1


def test_worker_evidence_that_crashes_its_verifier_keeps_the_exit_codes(
    evidence, tmp_path, capsys
):
    output = tmp_path / "export"
    assert main(export_argv(evidence, output)) == 0
    capsys.readouterr()
    worker = tmp_path / "worker"
    shutil.copytree(evidence[2], worker)
    rewrite(worker, lambda report: report["fault"].update(checkpoint_interval=0))

    argv = export_argv(evidence, tmp_path / "new")
    argv[argv.index("--worker-run-dir") + 1] = str(worker)
    assert main(argv) == 2
    assert "worker-recovery evidence was rejected" in capsys.readouterr().err
    assert not (tmp_path / "new").exists()
    argv = verify_argv(evidence, output)
    argv[argv.index("--worker-run-dir") + 1] = str(worker)
    assert main(argv) == 1
    assert "worker-recovery evidence was rejected" in capsys.readouterr().err


def test_usage_lists_both_commands_and_refuses_abbreviations(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith("usage: python -m ai_journey.telemetry_metrics_cli ")
    assert "{export,verify}" in usage
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "verify",
                "--output",
                "x",
                "--serving-run-dir",
                "y",
                "--worker-run-dir",
                "z",
            ]
        )
    assert stopped.value.code == 2
    with pytest.raises(SystemExit) as stopped:
        main(["scrape"])
    assert stopped.value.code == 2
