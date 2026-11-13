"""Internal subprocess entry point for checkpointed recovery workers."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .recovery_config import config_from_mapping
from .recovery_progress import WorkerProgress, write_worker_progress
from .recovery_worker import WorkerAttemptResult, run_worker_attempt
from .recovery_workload import build_recovery_workload


def _load_config(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("config must be a bounded regular non-symlink file")
    return config_from_mapping(json.loads(path.read_bytes()))


def execute_worker_process(
    *,
    source_path: Path,
    config_path: Path,
    run_id: str,
    attempt_id: str,
    checkpoint_path: Path,
    progress_path: Path,
    start_items: int,
    fault_after_items: int | None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    fault_wait: Callable[[], None] | None = None,
) -> WorkerAttemptResult:
    """Run one process attempt and expose a controller-observable fault boundary."""

    config = _load_config(config_path)
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("source must be a regular non-symlink file")
    source = source_path.read_bytes()
    workload = build_recovery_workload(source, config)
    if fault_after_items is not None and fault_after_items != config.fault_after_items:
        raise ValueError("fault_after_items must match the admitted config")
    wait = fault_wait or (lambda: threading.Event().wait())

    def observe(completed: int, accumulator: str) -> None:
        state = "fault_ready" if completed == fault_after_items else "running"
        if state == "fault_ready" or completed % config.heartbeat_interval_items == 0:
            write_worker_progress(
                progress_path,
                WorkerProgress(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    completed_items=completed,
                    accumulator_sha256=accumulator,
                    monotonic_ns=clock_ns(),
                    state=state,
                ),
            )
        if state == "fault_ready":
            wait()

    result = run_worker_attempt(
        workload=workload,
        config=config,
        run_id=run_id,
        attempt_id=attempt_id,
        checkpoint_path=checkpoint_path,
        start_items=start_items,
        on_item=observe,
    )
    write_worker_progress(
        progress_path,
        WorkerProgress(
            run_id=run_id,
            attempt_id=attempt_id,
            completed_items=result.completed_items,
            accumulator_sha256=result.accumulator_sha256,
            monotonic_ns=clock_ns(),
            state="completed",
        ),
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--start-items", type=int, default=0)
    parser.add_argument("--fault-after-items", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    execute_worker_process(
        source_path=arguments.source,
        config_path=arguments.config,
        run_id=arguments.run_id,
        attempt_id=arguments.attempt_id,
        checkpoint_path=arguments.checkpoint,
        progress_path=arguments.progress,
        start_items=arguments.start_items,
        fault_after_items=arguments.fault_after_items,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
