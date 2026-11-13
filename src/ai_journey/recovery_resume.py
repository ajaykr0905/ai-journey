"""Verified checkpoint admission for restarted recovery workers."""

from __future__ import annotations

from dataclasses import dataclass

from .recovery_accumulator import rebuild_accumulator
from .recovery_checkpoint import RecoveryCheckpoint
from .recovery_config import RecoveryExperimentConfig, RecoveryExperimentError
from .recovery_identity import recovery_attempt_id
from .recovery_progress import WorkerProgress
from .recovery_workload import RecoveryWorkload


@dataclass(frozen=True)
class RecoveryResumeDecision:
    """Exact durable boundary and identity admitted for restart."""

    start_items: int
    lost_uncheckpointed_items: int
    parent_checkpoint_sha256: str
    next_attempt_id: str


def admit_recovery_resume(
    checkpoint: RecoveryCheckpoint,
    failed_progress: WorkerProgress,
    workload: RecoveryWorkload,
    config: RecoveryExperimentConfig,
) -> RecoveryResumeDecision:
    """Admit restart only from a verified matching checkpoint prefix."""

    if not isinstance(checkpoint, RecoveryCheckpoint):
        raise TypeError("checkpoint must be RecoveryCheckpoint")
    if not isinstance(failed_progress, WorkerProgress):
        raise TypeError("failed_progress must be WorkerProgress")
    if not isinstance(workload, RecoveryWorkload):
        raise TypeError("workload must be RecoveryWorkload")
    if not isinstance(config, RecoveryExperimentConfig):
        raise TypeError("config must be RecoveryExperimentConfig")
    if checkpoint.run_id != failed_progress.run_id:
        raise RecoveryExperimentError("checkpoint and failed progress run IDs differ")
    if checkpoint.attempt_id != failed_progress.attempt_id:
        raise RecoveryExperimentError("checkpoint and failed progress attempts differ")
    if checkpoint.workload_sha256 != workload.fingerprint():
        raise RecoveryExperimentError("checkpoint workload identity does not match")
    if checkpoint.completed_items % config.checkpoint_interval != 0:
        raise RecoveryExperimentError("checkpoint is not on an admitted interval")
    if not checkpoint.completed_items < failed_progress.completed_items:
        raise RecoveryExperimentError("recovery requires lost uncheckpointed progress")
    if checkpoint.accumulator_sha256 != rebuild_accumulator(
        workload, checkpoint.completed_items
    ):
        raise RecoveryExperimentError(
            "checkpoint accumulator does not match its prefix"
        )
    parent_digest = checkpoint.to_mapping()["checkpoint_sha256"]
    attempt_number = int(checkpoint.attempt_id.split("-", 2)[1]) + 1
    return RecoveryResumeDecision(
        start_items=checkpoint.completed_items,
        lost_uncheckpointed_items=(
            failed_progress.completed_items - checkpoint.completed_items
        ),
        parent_checkpoint_sha256=parent_digest,
        next_attempt_id=recovery_attempt_id(
            checkpoint.run_id,
            attempt_number,
            parent_checkpoint_sha256=parent_digest,
        ),
    )
