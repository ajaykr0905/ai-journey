"""Supervise only owned expert replicas while leasing their mutable artifacts."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .checkpoint_journal import _read_bytes
from .ddp_launch import _owned_run, _paths_alias, WorkerFailure, WorkerTimeout
from .moe_ddp_plan import (
    MAX_REPORT_BYTES,
    read_public_moe_ddp_plan,
    unique_fields,
)


def launch_moe_ddp(
    *,
    config: Path,
    source: Path,
    checkpoint: Path,
    workdir: Path,
    steps: int,
    resume: bool = False,
    timeout_seconds: int = 60,
):
    plan, _ = read_public_moe_ddp_plan(config, source)
    if (
        type(steps) is not int
        or not 1 <= steps <= plan.run.optimizer_steps
        or type(resume) is not bool
        or type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= 300
    ):
        raise ValueError("expert launcher requires bounded steps/deadline and resume")
    if (
        checkpoint.is_symlink()
        or (checkpoint.exists() and not checkpoint.is_file())
        or (checkpoint.exists() != resume)
        or any(_paths_alias(checkpoint, path) for path in (source, config))
    ):
        raise ValueError("checkpoint must be a new output or an explicit resume input")
    if (
        workdir.is_symlink()
        or (workdir.exists() and not workdir.is_dir())
        or any(
            path.resolve().is_relative_to(workdir.resolve())
            for path in (config, source, checkpoint)
        )
    ):
        raise ValueError("worker directory must be disjoint from protected files")
    with (
        _owned_run(checkpoint, workdir),
        tempfile.TemporaryDirectory(prefix="ai-journey-moe-ddp-") as temporary,
    ):
        if any(workdir.iterdir()):
            raise ValueError(
                "worker directory must be empty; previous logs are retained"
            )
        processes, logs = [], []
        try:
            for rank in range(plan.run.world_size):
                log = open(workdir / f"rank-{rank}.log", "xb")
                logs.append(log)
                command = [
                    sys.executable,
                    "-m",
                    "ai_journey.moe_ddp_worker",
                    "--config",
                    str(config.resolve()),
                    "--source",
                    str(source.resolve()),
                    "--checkpoint",
                    str(checkpoint.resolve()),
                    "--output",
                    str((workdir / f"rank-{rank}.json").resolve()),
                    "--rank",
                    str(rank),
                    "--rendezvous",
                    (Path(temporary) / "rendezvous").as_uri(),
                    "--steps",
                    str(steps),
                ]
                if resume:
                    command.append("--resume")
                processes.append(
                    subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(
                            os.environ,
                            OMP_NUM_THREADS="1",
                            MKL_NUM_THREADS="1",
                            PYTHONUNBUFFERED="1",
                        ),
                    )
                )
            deadline = time.monotonic() + timeout_seconds
            while True:
                statuses = [process.poll() for process in processes]
                if any(status not in (None, 0) for status in statuses):
                    raise WorkerFailure(
                        "an expert worker failed; inspect retained logs"
                    )
                if all(status is not None for status in statuses):
                    break
                if time.monotonic() >= deadline:
                    raise WorkerTimeout("expert worker deadline exceeded")
                time.sleep(0.02)
            return [
                json.loads(
                    _read_bytes(workdir / f"rank-{rank}.json", MAX_REPORT_BYTES),
                    object_pairs_hook=unique_fields,
                )
                for rank in range(plan.run.world_size)
            ]
        finally:
            # Keep artifact leases through child reaping and log closure, also
            # when a later spawn/read/validation fails. Never kill caller PIDs.
            try:
                for process in processes:
                    if process.poll() is None:
                        try:
                            process.terminate()
                        except ProcessLookupError:
                            pass  # An owned child exited between poll and signal.
                for process in processes:
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=3)
            finally:
                for log in logs:
                    log.close()
