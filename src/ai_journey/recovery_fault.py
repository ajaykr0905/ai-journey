"""Explicit process-kill injection at a controller-observed work boundary."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .recovery_config import RecoveryExperimentConfig, RecoveryExperimentError
from .recovery_progress import WorkerProgress


class KillableProcess(Protocol):
    def poll(self) -> int | None: ...

    def kill(self) -> None: ...


@dataclass(frozen=True)
class FaultInjection:
    """Observed boundary and controller timestamp for one real process kill."""

    attempt_id: str
    completed_items: int
    injected_monotonic_ns: int
    mechanism: str = "process_kill"


def inject_process_kill(
    process: KillableProcess,
    progress: WorkerProgress,
    config: RecoveryExperimentConfig,
    *,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> FaultInjection:
    """Kill a live worker only at its admitted fault-ready boundary."""

    if not isinstance(progress, WorkerProgress):
        raise TypeError("progress must be WorkerProgress")
    if not isinstance(config, RecoveryExperimentConfig):
        raise TypeError("config must be RecoveryExperimentConfig")
    if progress.state != "fault_ready":
        raise RecoveryExperimentError(
            "worker must declare fault_ready before injection"
        )
    if progress.completed_items != config.fault_after_items:
        raise RecoveryExperimentError(
            "fault boundary does not match the admitted config"
        )
    if process.poll() is not None:
        raise RecoveryExperimentError("cannot inject a fault into an exited worker")
    injected = clock_ns()
    process.kill()
    return FaultInjection(
        attempt_id=progress.attempt_id,
        completed_items=progress.completed_items,
        injected_monotonic_ns=injected,
    )
