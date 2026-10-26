"""Bounded subprocess supervision for real local distributed workers."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .ddp_contracts import integer
from .ddp_limits import preflight


class WorkerFailure(RuntimeError):
    """A failed owned worker, with supervision observations after all children exit."""


class WorkerTimeout(TimeoutError):
    """An expired owned-worker deadline with the same supervision evidence."""


def _paths_alias(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and left.samefile(right)
    )


def launch_training(
    *,
    source: Path,
    spec: Path,
    checkpoint: Path,
    output_dir: Path,
    world_size: int = 2,
    steps: int = 2,
    resume: bool = False,
    timeout_seconds: int = 60,
    checkpoint_interval: int | None = None,
    failure_rank: int | None = None,
    failure_mode: str = "exit",
    failure_after_step: int | None = None,
    observe_recovery: bool = False,
) -> list[dict]:
    import json

    integer(world_size, "world_size", 1)
    integer(steps, "steps", 1)
    integer(timeout_seconds, "timeout_seconds", 1)
    if world_size > 4:
        raise ValueError("CPU lab permits at most four workers")
    training_spec = preflight(source, spec, world_size=world_size, steps=steps)
    if checkpoint_interval is not None:
        integer(checkpoint_interval, "checkpoint_interval", 1)
    if failure_after_step is not None:
        integer(failure_after_step, "failure_after_step", 1)
        if (
            failure_rank is None
            or failure_mode not in ("exit", "crash")
            or failure_after_step > training_spec.training.optimizer_steps
        ):
            raise ValueError(
                "late failure requires a rank, exit/crash mode and scheduled step"
            )
    if not isinstance(resume, bool):
        raise TypeError("resume must be boolean")
    if type(observe_recovery) is not bool or (observe_recovery and not resume):
        raise ValueError("recovery observations require an explicitly resumed run")
    if checkpoint.is_symlink() or (checkpoint.exists() and not checkpoint.is_file()):
        raise ValueError("checkpoint output must be a regular non-symlink path")
    if any(_paths_alias(checkpoint, path) for path in (source, spec)):
        raise ValueError("checkpoint output must not overwrite an input")
    protected = (source, spec, checkpoint)
    worker_outputs = [
        output_dir / f"rank-{rank}.{suffix}"
        for rank in range(world_size)
        for suffix in ("log", "json", "crash.json", "recovery.json")
    ]
    if any(
        path.is_symlink()
        or (path.exists() and not path.is_file())
        or any(_paths_alias(path, target) for target in protected)
        or any(_paths_alias(path, other) for other in worker_outputs[:index])
        for index, path in enumerate(worker_outputs)
    ):
        raise ValueError("worker outputs must not overwrite inputs or checkpoints")
    if failure_rank is not None:
        integer(failure_rank, "failure_rank")
        if failure_rank >= world_size or failure_mode not in (
            "exit",
            "crash",
            "stall",
            "nonfinite",
            "schedule",
        ):
            raise ValueError(
                "failure injection requires an existing rank and supported mode"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ai-journey-ddp-") as temporary:
        rendezvous = (Path(temporary) / "rendezvous").as_uri()
        processes = []
        logs = []
        exits = {}
        failure = None
        started_ns = time.monotonic_ns()
        try:
            for rank in range(world_size):
                log = open(output_dir / f"rank-{rank}.log", "wb")
                logs.append(log)
                command = [
                    sys.executable,
                    "-m",
                    "ai_journey.ddp_worker",
                    "--source",
                    str(source.resolve()),
                    "--spec",
                    str(spec.resolve()),
                    "--checkpoint",
                    str(checkpoint.resolve()),
                    "--output",
                    str((output_dir / f"rank-{rank}.json").resolve()),
                    "--rank",
                    str(rank),
                    "--world-size",
                    str(world_size),
                    "--rendezvous",
                    rendezvous,
                    "--steps",
                    str(steps),
                ]
                if resume:
                    command.append("--resume")
                if observe_recovery:
                    command.append("--observe-recovery")
                if checkpoint_interval is not None:
                    command.extend(["--checkpoint-interval", str(checkpoint_interval)])
                if rank == failure_rank:
                    command.extend(["--inject-failure", failure_mode])
                    if failure_after_step is not None:
                        command.extend(
                            ["--failure-after-step", str(failure_after_step)]
                        )
                environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                processes.append(
                    subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, env=environment
                    )
                )
            deadline = time.monotonic() + timeout_seconds
            while True:
                statuses = [process.poll() for process in processes]
                for rank, status in enumerate(statuses):
                    if status is not None:
                        exits.setdefault(rank, time.monotonic_ns())
                if any(status not in (None, 0) for status in statuses):
                    raise WorkerFailure(
                        "a distributed worker failed; inspect rank logs"
                    )
                if all(status is not None for status in statuses):
                    break
                if time.monotonic() >= deadline:
                    raise WorkerTimeout("distributed worker deadline exceeded")
                time.sleep(0.02)
            return [
                json.loads((output_dir / f"rank-{rank}.json").read_text())
                for rank in range(world_size)
            ]
        except (WorkerFailure, WorkerTimeout) as error:
            failure = error
            failure_observed_ns = time.monotonic_ns()
            raise
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            for log in logs:
                log.close()
            if failure is not None:
                reaped_ns = time.monotonic_ns()
                failure.supervision = {
                    "schema": "ai-journey-ddp-supervision-v1",
                    "started_ns": started_ns,
                    "failure_observed_ns": failure_observed_ns,
                    "reaped_ns": reaped_ns,
                    "timed_out": isinstance(failure, WorkerTimeout),
                    "workers": [
                        {
                            "rank": rank,
                            "pid": process.pid,
                            "returncode": process.returncode,
                            "exit_observed_ns": exits.get(rank, reaped_ns),
                        }
                        for rank, process in enumerate(processes)
                    ],
                }
