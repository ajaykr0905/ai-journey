from dataclasses import replace

import pytest
from test_recovery_config import valid_config

from ai_journey.recovery_accumulator import rebuild_accumulator
from ai_journey.recovery_checkpoint import RecoveryCheckpoint
from ai_journey.recovery_identity import recovery_attempt_id, recovery_run_id
from ai_journey.recovery_progress import WorkerProgress
from ai_journey.recovery_resume import admit_recovery_resume
from ai_journey.recovery_workload import build_recovery_workload


def evidence():
    config = valid_config()
    workload = build_recovery_workload(b"public recovery fixture", config)
    run = recovery_run_id(config)
    attempt = recovery_attempt_id(run, 1)
    checkpoint = RecoveryCheckpoint(
        run_id=run,
        attempt_id=attempt,
        completed_items=8,
        workload_sha256=workload.fingerprint(),
        accumulator_sha256=rebuild_accumulator(workload, 8),
    )
    progress = WorkerProgress(
        run_id=run,
        attempt_id=attempt,
        completed_items=11,
        accumulator_sha256=rebuild_accumulator(workload, 11),
        monotonic_ns=100,
        state="fault_ready",
    )
    return config, workload, checkpoint, progress


def test_resume_admits_exact_checkpoint_and_counts_lost_work():
    config, workload, checkpoint, progress = evidence()

    decision = admit_recovery_resume(checkpoint, progress, workload, config)

    assert decision.start_items == 8
    assert decision.lost_uncheckpointed_items == 3
    assert len(decision.parent_checkpoint_sha256) == 64
    assert decision.next_attempt_id.startswith("attempt-2-")


@pytest.mark.parametrize(
    ("checkpoint_change", "progress_change", "message"),
    [
        ({"run_id": "other-run-0123456789abcdef"}, {}, "run IDs"),
        ({}, {"attempt_id": "attempt-2-fedcba9876543210"}, "attempts"),
        ({"workload_sha256": "f" * 64}, {}, "workload"),
        ({"completed_items": 7}, {}, "interval"),
        ({"completed_items": 12}, {}, "lost uncheckpointed"),
        ({"accumulator_sha256": "f" * 64}, {}, "accumulator"),
    ],
)
def test_resume_rejects_mismatched_or_unproven_lineage(
    checkpoint_change, progress_change, message
):
    config, workload, checkpoint, progress = evidence()
    checkpoint = replace(checkpoint, **checkpoint_change)
    progress = replace(progress, **progress_change)

    with pytest.raises(ValueError, match=message):
        admit_recovery_resume(checkpoint, progress, workload, config)
