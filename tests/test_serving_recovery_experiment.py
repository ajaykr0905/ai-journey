import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest
from test_serving_recovery_config import valid_config

import ai_journey.serving_recovery_experiment as experiment
from ai_journey.serving_recovery_experiment import run_serving_recovery_experiment
from ai_journey.serving_recovery_io import load_attempt_log, load_trace

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "day-19-demo-names.txt"


def experiment_config(**changes):
    # An SLO equal to the request deadline makes recovery independent of host speed.
    return valid_config(**({"slo_latency_ms": 1000} | changes))


def inputs(tmp_path, **changes):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(experiment_config(**changes).to_mapping()))
    return FIXTURE, config


def run(tmp_path, **changes):
    source, config = inputs(tmp_path, **changes)
    return run_serving_recovery_experiment(
        source_path=source, config_path=config, run_dir=tmp_path / "run"
    )


def test_real_sigkill_under_load_is_detected_restarted_and_reconciled(tmp_path):
    result = run(tmp_path)
    config = experiment_config()
    run_dir = tmp_path / "run"

    assert result.fault.acknowledged_responses == config.fault_after_acknowledged
    assert (result.fault.lifetime_number, result.fault.signal) == (1, "SIGKILL")
    assert [event.kind for event in result.events] == [
        "spawn",
        "ready",
        "fault",
        "exit_observed",
        "restart",
        "ready",
        "recovered",
        "finished",
    ]
    assert result.events[3].payload["exit_code"] == -signal.SIGKILL
    assert result.events[-1].payload["exit_code"] == 0
    totals = result.reconciliation.totals
    assert (totals["requests"], totals["succeeded"], totals["failed"]) == (24, 24, 0)
    assert totals["success_after_retry"] >= 1
    assert totals["executions"] == 24 + totals["lost_response_executions"]
    failed = {a.outcome for a in result.attempts if a.outcome != "ok"}
    assert failed and failed <= {"connection_refused", "connection_reset", "timeout"}
    first, second = result.journals
    assert len(first.records) >= config.fault_after_acknowledged
    assert len(first.records) + len(second.records) == totals["executions"]
    assert second.torn_tail_bytes == 0

    timing = result.timing
    assert timing.detection_latency_ns >= 0
    assert timing.restart_latency_ns > 0
    assert timing.unavailability_ns is not None and timing.unavailability_ns > 0
    assert timing.time_to_slo_recovery_ns >= timing.detection_latency_ns
    assert timing.pre_fault_latency.samples >= 1
    assert timing.post_fault_latency.samples >= config.slo_consecutive_successes
    assert result.run_duration_ns <= config.max_run_ms * 1_000_000

    assert load_attempt_log(run_dir / "attempts.json") == result.attempts
    records, _ = load_trace(run_dir / "trace.json")
    assert records[-1]["event_sha256"] == result.trace_sha256
    assert (run_dir / "source.txt").read_bytes() == FIXTURE.read_bytes()
    assert json.loads((run_dir / "config.json").read_bytes()) == config.to_mapping()
    for event in result.events:
        if "pid" in event.payload:
            with pytest.raises(ProcessLookupError):
                os.kill(event.payload["pid"], 0)


def test_experiment_refuses_an_existing_run_directory(tmp_path):
    source, config = inputs(tmp_path)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "owned.txt").write_text("keep")

    with pytest.raises(ValueError, match="must not already exist"):
        run_serving_recovery_experiment(
            source_path=source, config_path=config, run_dir=tmp_path / "run"
        )
    assert (tmp_path / "run" / "owned.txt").read_text() == "keep"


def test_a_server_that_cannot_start_fails_the_run_without_orphans(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        experiment,
        "server_command",
        lambda **_: (sys.executable, "-I", "-c", "raise SystemExit(5)"),
    )
    with pytest.raises(ValueError, match="code 5"):
        run(tmp_path)
    assert sorted(path.name for path in (tmp_path / "run").iterdir()) == [
        "config.json",
        "lifetime-1.stderr",
        "source.txt",
    ]


@pytest.mark.parametrize(
    ("replacement", "message", "stderr_logs"),
    [
        # Popen rejects a NUL byte with ValueError, which no restart expects.
        (("-c", "\0"), "monitor failed: ValueError: embedded null", 1),
        (("-c", "raise SystemExit(4)"), "restart limit of 2 exhausted", 3),
    ],
)
def test_a_failed_supervisor_ends_the_run_promptly_with_its_error(
    tmp_path, monkeypatch, replacement, message, stderr_logs
):
    real = experiment.server_command

    def command(**options):
        if options["journal_path"].name == "journal-lifetime-1.jsonl":
            return real(**options)
        return (sys.executable, "-I", *replacement)

    monkeypatch.setattr(experiment, "server_command", command)
    max_run_ms = 20_000
    started = time.monotonic()
    with pytest.raises(ValueError, match=message):
        run(tmp_path, max_run_ms=max_run_ms)
    assert time.monotonic() - started < max_run_ms / 1000 / 2
    assert sorted(path.name for path in (tmp_path / "run").iterdir()) == [
        "config.json",
        "journal-lifetime-1.jsonl",
        *(f"lifetime-{number}.stderr" for number in range(1, stderr_logs + 1)),
        "source.txt",
    ]
