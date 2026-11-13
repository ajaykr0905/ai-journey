import pytest
from test_recovery_config import valid_config

from ai_journey.recovery_accumulator import rebuild_accumulator
from ai_journey.recovery_checkpoint_io import load_recovery_checkpoint
from ai_journey.recovery_identity import recovery_attempt_id, recovery_run_id
from ai_journey.recovery_worker import run_worker_attempt
from ai_journey.recovery_workload import build_recovery_workload


def inputs():
    config = valid_config()
    workload = build_recovery_workload(b"public recovery fixture", config)
    run = recovery_run_id(config)
    return config, workload, run, recovery_attempt_id(run, 1)


def test_worker_processes_all_items_and_publishes_terminal_checkpoint(tmp_path):
    config, workload, run, attempt = inputs()
    observed = []

    result = run_worker_attempt(
        workload=workload,
        config=config,
        run_id=run,
        attempt_id=attempt,
        checkpoint_path=tmp_path / "checkpoint.json",
        on_item=lambda completed, digest: observed.append((completed, digest)),
    )

    checkpoint = load_recovery_checkpoint(tmp_path / "checkpoint.json")
    assert result.started_items == 0
    assert result.completed_items == config.work_items
    assert result.last_checkpoint_items == config.work_items
    assert result.accumulator_sha256 == rebuild_accumulator(workload, 20)
    assert checkpoint.completed_items == 20
    assert checkpoint.accumulator_sha256 == result.accumulator_sha256
    assert [item[0] for item in observed] == list(range(1, 21))


def test_worker_resumes_from_prefix_without_reprocessing_prior_items(tmp_path):
    config, workload, run, _ = inputs()
    attempt = recovery_attempt_id(run, 2, parent_checkpoint_sha256="a" * 64)
    observed = []

    result = run_worker_attempt(
        workload=workload,
        config=config,
        run_id=run,
        attempt_id=attempt,
        checkpoint_path=tmp_path / "checkpoint.json",
        start_items=8,
        on_item=lambda completed, _digest: observed.append(completed),
    )

    assert result.started_items == 8
    assert observed == list(range(9, 21))
    assert result.accumulator_sha256 == rebuild_accumulator(workload, 20)


@pytest.mark.parametrize("start", [True, -1, 20])
def test_worker_rejects_invalid_resume_boundaries(tmp_path, start):
    config, workload, run, attempt = inputs()
    with pytest.raises((TypeError, ValueError), match="integer|precede"):
        run_worker_attempt(
            workload=workload,
            config=config,
            run_id=run,
            attempt_id=attempt,
            checkpoint_path=tmp_path / "checkpoint.json",
            start_items=start,
        )
