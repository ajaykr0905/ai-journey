import json

import pytest
from test_recovery_config import valid_config

from ai_journey.recovery_checkpoint_io import load_recovery_checkpoint
from ai_journey.recovery_identity import recovery_attempt_id, recovery_run_id
from ai_journey.recovery_progress import load_worker_progress
from ai_journey.recovery_worker_cli import execute_worker_process, main


def inputs(tmp_path):
    config = valid_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_mapping()))
    source_path = tmp_path / "source.txt"
    source_path.write_bytes(b"public recovery fixture")
    run = recovery_run_id(config)
    attempt = recovery_attempt_id(run, 1)
    return config, config_path, source_path, run, attempt


def test_process_entry_point_completes_and_publishes_progress(tmp_path):
    config, config_path, source, run, attempt = inputs(tmp_path)
    ticks = iter(range(100, 1000))

    result = execute_worker_process(
        source_path=source,
        config_path=config_path,
        run_id=run,
        attempt_id=attempt,
        checkpoint_path=tmp_path / "checkpoint.json",
        progress_path=tmp_path / "progress.json",
        start_items=0,
        fault_after_items=None,
        clock_ns=lambda: next(ticks),
    )

    progress = load_worker_progress(tmp_path / "progress.json")
    checkpoint = load_recovery_checkpoint(tmp_path / "checkpoint.json")
    assert result.completed_items == config.work_items
    assert progress.state == "completed"
    assert progress.completed_items == config.work_items
    assert checkpoint.completed_items == config.work_items


def test_process_entry_point_exposes_fault_boundary_before_checkpoint(tmp_path):
    config, config_path, source, run, attempt = inputs(tmp_path)

    def stop():
        raise RuntimeError("test stop at fault boundary")

    with pytest.raises(RuntimeError, match="fault boundary"):
        execute_worker_process(
            source_path=source,
            config_path=config_path,
            run_id=run,
            attempt_id=attempt,
            checkpoint_path=tmp_path / "checkpoint.json",
            progress_path=tmp_path / "progress.json",
            start_items=0,
            fault_after_items=config.fault_after_items,
            clock_ns=lambda: 100,
            fault_wait=stop,
        )

    progress = load_worker_progress(tmp_path / "progress.json")
    checkpoint = load_recovery_checkpoint(tmp_path / "checkpoint.json")
    assert progress.state == "fault_ready"
    assert progress.completed_items == config.fault_after_items
    assert checkpoint.completed_items == 8


def test_internal_cli_completes_normal_attempt(tmp_path):
    _, config_path, source, run, attempt = inputs(tmp_path)
    assert (
        main(
            [
                "--source",
                str(source),
                "--config",
                str(config_path),
                "--run-id",
                run,
                "--attempt-id",
                attempt,
                "--checkpoint",
                str(tmp_path / "checkpoint.json"),
                "--progress",
                str(tmp_path / "progress.json"),
            ]
        )
        == 0
    )
