import json

import pytest
from test_recovery_config import valid_config

import ai_journey.recovery_experiment as experiment
from ai_journey.recovery_config import RecoveryExperimentError
from ai_journey.recovery_experiment import run_recovery_experiment
from ai_journey.recovery_progress import load_worker_progress
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


def observe_exit_code(monkeypatch, worker, exit_code):
    """Report ``exit_code`` once the given worker (0 or 1) has really exited."""

    workers = []
    real_start = experiment.start_worker_process
    real_wait = experiment.wait_for_process_exit

    def start(**options):
        workers.append(real_start(**options))
        return workers[-1]

    def wait(process, **options):
        observed, detected_ns = real_wait(process, **options)
        if workers.index(process) == worker:
            observed = exit_code
        return observed, detected_ns

    monkeypatch.setattr(experiment, "start_worker_process", start)
    monkeypatch.setattr(experiment, "wait_for_process_exit", wait)
    return workers


@pytest.mark.parametrize(
    ("worker", "exit_code", "message", "kinds"),
    [
        (
            0,
            0,
            "faulted worker exited successfully",
            ["started", "fault_injected"],
        ),
        (
            1,
            1,
            "resumed worker did not exit successfully",
            ["started", "fault_injected", "failure_detected", "resumed"],
        ),
    ],
)
def test_worker_exit_code_guards_stop_the_experiment_and_leave_no_worker(
    tmp_path, monkeypatch, worker, exit_code, message, kinds
):
    source, config = inputs(tmp_path)
    run_dir = tmp_path / "run"
    workers = observe_exit_code(monkeypatch, worker, exit_code)

    try:
        with pytest.raises(RecoveryExperimentError, match=message):
            run_recovery_experiment(
                source_path=source, config_path=config, run_dir=run_dir
            )
        # The trace stops early, so it is read raw rather than as a finished trace.
        trace = json.loads((run_dir / "trace.json").read_bytes())
        assert [event["kind"] for event in trace] == kinds
        assert len(workers) == worker + 1
        assert all(process.poll() is not None for process in workers)
        if worker == 1:
            progress = load_worker_progress(run_dir / "progress.json")
            assert progress.state == "completed"
    finally:
        for process in workers:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
