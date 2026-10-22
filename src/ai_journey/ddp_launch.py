"""Bounded subprocess supervision for real local distributed workers."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .ddp_contracts import integer
from .ddp_limits import preflight


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
    failure_rank: int | None = None,
    failure_mode: str = "exit",
) -> list[dict]:
    import json

    integer(world_size, "world_size", 1)
    integer(steps, "steps", 1)
    integer(timeout_seconds, "timeout_seconds", 1)
    if world_size > 4:
        raise ValueError("CPU lab permits at most four workers")
    preflight(source, spec, world_size=world_size, steps=steps)
    if not isinstance(resume, bool):
        raise TypeError("resume must be boolean")
    if checkpoint.resolve() in (source.resolve(), spec.resolve()):
        raise ValueError("checkpoint output must not overwrite an input")
    if failure_rank is not None:
        integer(failure_rank, "failure_rank")
        if failure_rank >= world_size or failure_mode not in (
            "exit",
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
                if rank == failure_rank:
                    command.extend(["--inject-failure", failure_mode])
                environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                processes.append(
                    subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, env=environment
                    )
                )
            deadline = time.monotonic() + timeout_seconds
            while any(process.poll() is None for process in processes):
                if any(process.poll() not in (None, 0) for process in processes):
                    raise RuntimeError("a distributed worker failed; inspect rank logs")
                if time.monotonic() >= deadline:
                    raise TimeoutError("distributed worker deadline exceeded")
                time.sleep(0.02)
            if any(process.returncode != 0 for process in processes):
                raise RuntimeError("a distributed worker failed; inspect rank logs")
            return [
                json.loads((output_dir / f"rank-{rank}.json").read_text())
                for rank in range(world_size)
            ]
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
