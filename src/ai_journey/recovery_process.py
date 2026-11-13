"""Shell-free subprocess launch controls for recovery workers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .recovery_config import RecoveryExperimentError


def worker_command(
    *,
    source_path: Path,
    config_path: Path,
    run_id: str,
    attempt_id: str,
    checkpoint_path: Path,
    progress_path: Path,
    start_items: int,
    fault_after_items: int | None,
) -> tuple[str, ...]:
    """Build an argument-vector command without shell interpolation."""

    paths = {
        "source_path": source_path,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "progress_path": progress_path,
    }
    if any(not isinstance(path, Path) for path in paths.values()):
        raise TypeError("worker paths must be pathlib.Path values")
    if isinstance(start_items, bool) or not isinstance(start_items, int):
        raise TypeError("start_items must be an integer")
    if start_items < 0:
        raise RecoveryExperimentError("start_items must be non-negative")
    command = [
        sys.executable,
        "-m",
        "ai_journey.recovery_worker_cli",
        "--source",
        str(source_path),
        "--config",
        str(config_path),
        "--run-id",
        run_id,
        "--attempt-id",
        attempt_id,
        "--checkpoint",
        str(checkpoint_path),
        "--progress",
        str(progress_path),
        "--start-items",
        str(start_items),
    ]
    if fault_after_items is not None:
        if isinstance(fault_after_items, bool) or not isinstance(
            fault_after_items, int
        ):
            raise TypeError("fault_after_items must be an integer")
        if fault_after_items <= start_items:
            raise RecoveryExperimentError(
                "fault_after_items must follow the starting boundary"
            )
        command.extend(["--fault-after-items", str(fault_after_items)])
    return tuple(command)


def start_worker_process(**options) -> subprocess.Popen[bytes]:
    """Launch a worker with no shell, stdin, or inherited output streams."""

    command = worker_command(**options)
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
    )
