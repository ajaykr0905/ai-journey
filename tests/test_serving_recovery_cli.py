import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.serving_recovery_cli import main
from ai_journey.serving_recovery_config import load_serving_recovery_config

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "day-19-demo-names.txt"
PINNED = ROOT / "config" / "day-70-serving-recovery.json"
DOC = ROOT / "docs" / "serving-failure-recovery.md"


@pytest.fixture(scope="module")
def pinned_run(tmp_path_factory):
    """Run the pinned experiment once through ``main`` for this module."""

    run_dir = tmp_path_factory.mktemp("serving-cli") / "run"
    argv = ["run", "--source", str(SOURCE), "--config", str(PINNED)]
    argv += ["--run-dir", str(run_dir)]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = main(argv)
    return run_dir, argv, code, output.getvalue()


def test_pinned_config_is_strict_and_bounded():
    config = load_serving_recovery_config(PINNED)

    assert config.experiment_name == "day-70-serving-recovery"
    assert (config.requests, config.concurrency) == (48, 2)
    assert config.fault_after_acknowledged == 16
    assert config.retry_window_ms >= config.recovery_bound_ms
    assert config.max_run_ms == 60_000


def test_cli_kills_restarts_and_verifies_the_pinned_experiment(pinned_run, capsys):
    run_dir, _, code, output = pinned_run

    assert code == 0
    receipt = json.loads(output)
    assert receipt["verified"] is True and receipt["recovered"] is True
    assert receipt["requests"] == 48
    assert set(receipt["summary_ms"]) == {
        "detection_latency_ms",
        "restart_latency_ms",
        "time_to_slo_recovery_ms",
        "unavailability_ms",
    }
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "attempts.json",
        "config.json",
        "journal-lifetime-1.jsonl",
        "journal-lifetime-2.jsonl",
        "lifetime-1.stderr",
        "lifetime-2.stderr",
        "report.json",
        "source.txt",
        "trace.json",
    ]
    assert main(["verify", "--run-dir", str(run_dir)]) == 0
    assert json.loads(capsys.readouterr().out) == receipt

    module = subprocess.run(
        [sys.executable, "-m", "ai_journey.serving_recovery_cli", "verify"]
        + ["--run-dir", str(run_dir)],
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=60,
    )
    assert module.returncode == 0, module.stderr
    assert json.loads(module.stdout) == receipt


def test_cli_exit_codes_separate_invalid_runs_and_rejected_evidence(
    pinned_run, tmp_path, capsys
):
    run_dir, argv, _, _ = pinned_run

    assert main(argv) == 2
    assert "must not already exist" in capsys.readouterr().err
    broken = tmp_path / "config.json"
    broken.write_text('{"schema_version": 1, "schema_version": 1}')
    assert main(argv[:4] + [str(broken), "--run-dir", str(tmp_path / "new")]) == 2
    assert "duplicate JSON key" in capsys.readouterr().err
    assert not (tmp_path / "new").exists()

    copied = tmp_path / "copied"
    copied.mkdir()
    for path in run_dir.iterdir():
        (copied / path.name).write_bytes(path.read_bytes())
    (copied / "attempts.json").write_bytes(b'{"attempts":[],"schema_version":1}')
    assert main(["verify", "--run-dir", str(copied)]) == 1
    assert "verification failed" in capsys.readouterr().err


def test_operator_doc_states_commands_measurements_and_boundary():
    doc = DOC.read_text()

    for phrase in (
        "python -m ai_journey.serving_recovery_cli run",
        "python -m ai_journey.serving_recovery_cli verify",
        "config/day-70-serving-recovery.json",
        "Detection latency",
        "Restart latency",
        "Client-observed unavailability",
        "Time to SLO recovery",
        "nearest-rank",
        "at-least-once",
        "not GPU, CUDA, distributed",
        "Limitations",
    ):
        assert phrase in doc
