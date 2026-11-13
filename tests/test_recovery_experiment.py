import json

import pytest
from test_recovery_config import valid_config

from ai_journey.recovery_experiment import run_recovery_experiment
from ai_journey.recovery_trace_io import load_recovery_trace


def inputs(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"public recovery fixture")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(valid_config().to_mapping()))
    return source, config


def test_real_process_kill_recovers_exact_workload_and_measures_time(tmp_path):
    source, config = inputs(tmp_path)
    run_dir = tmp_path / "run"

    result = run_recovery_experiment(
        source_path=source, config_path=config, run_dir=run_dir
    )

    assert result.failed_exit_code != 0
    assert result.resumed_exit_code == 0
    assert result.failed_attempt_id != result.resumed_attempt_id
    assert result.loss.lost_uncheckpointed_items == 3
    assert result.loss.replayed_items == 3
    assert result.loss.omitted_items == 0
    assert result.loss.duplicated_committed_items == 0
    assert result.timing.detection_time_ms >= 0
    assert result.timing.recovery_time_ms > 0
    assert len(result.trace_sha256) == 64
    trace = load_recovery_trace(run_dir / "trace.json")
    assert [event.kind for event in trace] == [
        "started",
        "fault_injected",
        "failure_detected",
        "resumed",
        "completed",
    ]


def test_experiment_refuses_existing_destination(tmp_path):
    source, config = inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    marker = run_dir / "owned.txt"
    marker.write_text("preserve")

    with pytest.raises(ValueError, match="must not already exist"):
        run_recovery_experiment(source_path=source, config_path=config, run_dir=run_dir)
    assert marker.read_text() == "preserve"
