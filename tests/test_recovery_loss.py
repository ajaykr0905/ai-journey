from dataclasses import replace

import pytest
from test_recovery_resume import evidence

from ai_journey.recovery_accumulator import rebuild_accumulator
from ai_journey.recovery_loss import assess_recovery_loss
from ai_journey.recovery_progress import WorkerProgress


def completed_progress():
    config, workload, checkpoint, failed = evidence()
    return (
        config,
        workload,
        checkpoint,
        failed,
        WorkerProgress(
            run_id=failed.run_id,
            attempt_id="attempt-2-fedcba9876543210",
            completed_items=config.work_items,
            accumulator_sha256=rebuild_accumulator(workload, config.work_items),
            monotonic_ns=200,
            state="completed",
        ),
    )


def test_loss_assessment_reconciles_rollback_and_exact_final_result():
    config, workload, checkpoint, failed, final = completed_progress()

    assessment = assess_recovery_loss(
        failed_progress=failed,
        recovery_checkpoint=checkpoint,
        final_progress=final,
        workload=workload,
        config=config,
    )

    assert assessment.fault_observed_items == 11
    assert assessment.durable_items_at_fault == 8
    assert assessment.lost_uncheckpointed_items == 3
    assert assessment.replayed_items == 3
    assert assessment.final_items == 20
    assert assessment.omitted_items == 0
    assert assessment.duplicated_committed_items == 0


@pytest.mark.parametrize(
    ("failed_change", "final_change", "message"),
    [
        ({"run_id": "other-run-0123456789abcdef"}, {}, "run identities"),
        ({}, {"attempt_id": "same"}, "resumed"),
        ({"completed_items": 10}, {}, "fault boundary"),
        ({}, {"state": "running"}, "complete"),
        ({}, {"completed_items": 19}, "complete"),
        ({}, {"accumulator_sha256": "f" * 64}, "final result"),
    ],
)
def test_loss_assessment_rejects_unreconciled_evidence(
    failed_change, final_change, message
):
    config, workload, checkpoint, failed, final = completed_progress()
    failed = replace(failed, **failed_change)
    if final_change.get("attempt_id") == "same":
        final_change = {**final_change, "attempt_id": failed.attempt_id}
    final = replace(final, **final_change)
    with pytest.raises(ValueError, match=message):
        assess_recovery_loss(
            failed_progress=failed,
            recovery_checkpoint=checkpoint,
            final_progress=final,
            workload=workload,
            config=config,
        )
