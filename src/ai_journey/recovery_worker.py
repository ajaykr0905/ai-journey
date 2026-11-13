"""Checkpointed deterministic worker for recovery experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .recovery_accumulator import accumulate_work_item, rebuild_accumulator
from .recovery_checkpoint import RecoveryCheckpoint
from .recovery_checkpoint_io import write_recovery_checkpoint
from .recovery_config import RecoveryExperimentConfig, RecoveryExperimentError
from .recovery_workload import RecoveryWorkload


@dataclass(frozen=True)
class WorkerAttemptResult:
    """Final in-process state for a bounded worker attempt."""

    started_items: int
    completed_items: int
    accumulator_sha256: str
    last_checkpoint_items: int


def run_worker_attempt(
    *,
    workload: RecoveryWorkload,
    config: RecoveryExperimentConfig,
    run_id: str,
    attempt_id: str,
    checkpoint_path: Path,
    start_items: int = 0,
    on_item: Callable[[int, str], None] | None = None,
) -> WorkerAttemptResult:
    """Process remaining items and publish periodic plus terminal checkpoints."""

    if not isinstance(workload, RecoveryWorkload):
        raise TypeError("workload must be RecoveryWorkload")
    if not isinstance(config, RecoveryExperimentConfig):
        raise TypeError("config must be RecoveryExperimentConfig")
    if len(workload.items) != config.work_items:
        raise RecoveryExperimentError("workload length does not match config")
    if isinstance(start_items, bool) or not isinstance(start_items, int):
        raise TypeError("start_items must be an integer")
    if not 0 <= start_items < config.work_items:
        raise RecoveryExperimentError("start_items must precede workload completion")
    if on_item is not None and not callable(on_item):
        raise TypeError("on_item must be callable")
    accumulator = rebuild_accumulator(workload, start_items)
    last_checkpoint = start_items
    for item in workload.items[start_items:]:
        accumulator = accumulate_work_item(accumulator, item)
        completed = item.index + 1
        if on_item is not None:
            on_item(completed, accumulator)
        if (
            completed % config.checkpoint_interval == 0
            or completed == config.work_items
        ):
            write_recovery_checkpoint(
                checkpoint_path,
                RecoveryCheckpoint(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    completed_items=completed,
                    workload_sha256=workload.fingerprint(),
                    accumulator_sha256=accumulator,
                ),
            )
            last_checkpoint = completed
    return WorkerAttemptResult(
        started_items=start_items,
        completed_items=config.work_items,
        accumulator_sha256=accumulator,
        last_checkpoint_items=last_checkpoint,
    )
