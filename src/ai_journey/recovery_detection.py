"""Bounded controller polling for worker progress and process exit."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .recovery_config import RecoveryExperimentError
from .recovery_progress import WorkerProgress, load_worker_progress


class PollableProcess(Protocol):
    def poll(self) -> int | None: ...


def _progress_in_state(progress_path: Path, state: str) -> WorkerProgress | None:
    if progress_path.is_file() and not progress_path.is_symlink():
        progress = load_worker_progress(progress_path)
        if progress.state == state:
            return progress
    return None


def wait_for_progress_state(
    process: PollableProcess,
    progress_path: Path,
    expected_state: str,
    *,
    timeout_ms: int,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_ms: int = 5,
) -> WorkerProgress:
    """Wait for one declared state while failing fast on worker exit.

    A worker can publish ``expected_state`` and exit between a progress read
    and ``poll()``, so progress is read once more after an observed exit; the
    exit is an error only if that read does not show ``expected_state`` either.
    """

    if expected_state not in {"running", "fault_ready", "completed"}:
        raise RecoveryExperimentError("expected_state is unsupported")
    for name, value in (
        ("timeout_ms", timeout_ms),
        ("poll_interval_ms", poll_interval_ms),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise RecoveryExperimentError(f"{name} must be positive")
    started_ns = clock_ns()
    deadline_ns = started_ns + timeout_ms * 1_000_000
    while clock_ns() <= deadline_ns:
        progress = _progress_in_state(progress_path, expected_state)
        if progress is not None:
            return progress
        exit_code = process.poll()
        if exit_code is not None:
            # Every write the worker made before exiting is visible now.
            progress = _progress_in_state(progress_path, expected_state)
            if progress is not None:
                return progress
            raise RecoveryExperimentError(
                f"worker exited with code {exit_code} before {expected_state}"
            )
        sleep(poll_interval_ms / 1000)
    raise RecoveryExperimentError(
        f"worker did not reach {expected_state} before timeout"
    )


def wait_for_process_exit(
    process: PollableProcess,
    *,
    timeout_ms: int,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_ms: int = 5,
) -> tuple[int, int]:
    """Return the observed exit code and monotonic detection timestamp."""

    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int):
        raise TypeError("timeout_ms must be an integer")
    if timeout_ms <= 0:
        raise RecoveryExperimentError("timeout_ms must be positive")
    deadline_ns = clock_ns() + timeout_ms * 1_000_000
    while clock_ns() <= deadline_ns:
        exit_code = process.poll()
        if exit_code is not None:
            return exit_code, clock_ns()
        sleep(poll_interval_ms / 1000)
    raise RecoveryExperimentError("worker did not exit before timeout")
