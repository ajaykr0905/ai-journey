"""Exact committed, lost, replayed, duplicated, and omitted work accounting."""

from __future__ import annotations

from dataclasses import dataclass

from .recovery_accumulator import rebuild_accumulator
from .recovery_checkpoint import RecoveryCheckpoint
from .recovery_config import RecoveryExperimentConfig, RecoveryExperimentError
from .recovery_progress import WorkerProgress
from .recovery_workload import RecoveryWorkload


@dataclass(frozen=True)
class RecoveryLossAssessment:
    """Exact item accounting across the failed and resumed attempts."""

    fault_observed_items: int
    durable_items_at_fault: int
    lost_uncheckpointed_items: int
    replayed_items: int
    final_items: int
    omitted_items: int
    duplicated_committed_items: int


def assess_recovery_loss(
    *,
    failed_progress: WorkerProgress,
    recovery_checkpoint: RecoveryCheckpoint,
    final_progress: WorkerProgress,
    workload: RecoveryWorkload,
    config: RecoveryExperimentConfig,
) -> RecoveryLossAssessment:
    """Reconcile attempt boundaries against the complete deterministic result."""

    if any(
        not isinstance(item, expected)
        for item, expected in (
            (failed_progress, WorkerProgress),
            (recovery_checkpoint, RecoveryCheckpoint),
            (final_progress, WorkerProgress),
            (workload, RecoveryWorkload),
            (config, RecoveryExperimentConfig),
        )
    ):
        raise TypeError("loss assessment inputs have invalid types")
    if (
        len({failed_progress.run_id, recovery_checkpoint.run_id, final_progress.run_id})
        != 1
    ):
        raise RecoveryExperimentError("loss assessment run identities differ")
    if failed_progress.attempt_id != recovery_checkpoint.attempt_id:
        raise RecoveryExperimentError("failed progress and checkpoint attempts differ")
    if final_progress.attempt_id == failed_progress.attempt_id:
        raise RecoveryExperimentError("final progress must come from a resumed attempt")
    if failed_progress.completed_items != config.fault_after_items:
        raise RecoveryExperimentError(
            "failed progress does not match the fault boundary"
        )
    if final_progress.state != "completed" or final_progress.completed_items != len(
        workload.items
    ):
        raise RecoveryExperimentError("final progress does not complete the workload")
    expected = rebuild_accumulator(workload, len(workload.items))
    if final_progress.accumulator_sha256 != expected:
        raise RecoveryExperimentError(
            "final result does not match the complete workload"
        )
    lost = failed_progress.completed_items - recovery_checkpoint.completed_items
    if lost <= 0:
        raise RecoveryExperimentError("recovery did not expose uncheckpointed loss")
    return RecoveryLossAssessment(
        fault_observed_items=failed_progress.completed_items,
        durable_items_at_fault=recovery_checkpoint.completed_items,
        lost_uncheckpointed_items=lost,
        replayed_items=lost,
        final_items=final_progress.completed_items,
        omitted_items=0,
        duplicated_committed_items=0,
    )
