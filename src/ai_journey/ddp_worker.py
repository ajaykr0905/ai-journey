"""One bounded CPU/Gloo worker, launched only by the local lab supervisor."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from torch.nn.parallel import DistributedDataParallel

from .ddp_checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_checkpoint_journal,
    checkpoint_run_sha256,
)
from .checkpoint_journal import CheckpointJournal
from .ddp_contracts import Topology, integer
from .ddp_runtime import process_group, require_all_ranks
from .ddp_training import train_segment
from .gpt2_training import build_gpt2_optimizer
from .ddp_limits import preflight
from .training_models import build_training_model, training_model_fingerprint


def _write_boundary(path: Path, marker):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(marker, sort_keys=True, allow_nan=False).encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _abrupt_exit(output: Path, topology: Topology, *, completed_step: int):
    """Persist a public-safe boundary witness, then exit only this owned child."""
    _write_boundary(
        output.with_suffix(".crash.json"),
        {
            "schema": "ai-journey-ddp-crash-boundary-v1",
            "rank": topology.rank,
            "world_size": topology.world_size,
            "pid": os.getpid(),
            "completed_step": completed_step,
            "observed_ns": time.monotonic_ns(),
            "exit_code": 86,
        },
    )
    os._exit(86)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "spec", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rendezvous", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--observe-recovery", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--journal-directory", type=Path)
    parser.add_argument("--failure-after-step", type=int)
    parser.add_argument(
        "--inject-failure", choices=("exit", "crash", "stall", "nonfinite", "schedule")
    )
    args = parser.parse_args(argv)
    if args.journal_directory is not None and any(
        path.resolve().is_relative_to(args.journal_directory.resolve())
        for path in (args.source, args.spec, args.checkpoint, args.output)
    ):
        raise ValueError("journal directory must not contain protected worker paths")
    if args.observe_recovery and not args.resume:
        raise ValueError("recovery observations require a resumed worker")
    if args.checkpoint_interval is not None:
        integer(args.checkpoint_interval, "checkpoint_interval", 1)
    if args.failure_after_step is not None:
        integer(args.failure_after_step, "failure_after_step", 1)
        if args.inject_failure not in ("exit", "crash"):
            raise ValueError("late failure supports exit/crash modes only")
    if args.inject_failure == "exit" and args.failure_after_step is None:
        raise RuntimeError("injected worker exit before process-group initialization")
    if args.inject_failure == "stall":
        time.sleep(30)
    topology = Topology(args.rank, args.world_size, args.rank)
    if args.inject_failure == "crash" and args.failure_after_step is None:
        _abrupt_exit(args.output, topology, completed_step=0)
    spec = preflight(
        args.source, args.spec, world_size=args.world_size, steps=args.steps
    )
    source = args.source.read_bytes()
    from hashlib import sha256

    if sha256(source).hexdigest() != spec.source.sha256:
        raise ValueError("public source digest does not match specification")
    torch.set_num_threads(1)
    torch.manual_seed(spec.training.seed)
    model = build_training_model(spec.model, max_parameters=1_000_000).double()
    optimizer = build_gpt2_optimizer(model, spec.training)
    with process_group(topology, rendezvous=args.rendezvous):
        wrapped = DistributedDataParallel(model)
        if args.journal_directory is not None and not args.resume:
            journal_ready = True
            if topology.rank == 0:
                try:
                    journal = CheckpointJournal(
                        args.journal_directory,
                        run_sha256=checkpoint_run_sha256(
                            model, spec.training, source, optimizer=optimizer
                        ),
                    )
                    journal_ready = journal.recorded_step is None
                except (OSError, ValueError, RuntimeError, TypeError):
                    journal_ready = False
            require_all_ranks(journal_ready, phase="new checkpoint journal admission")
        if args.inject_failure == "nonfinite":
            model.wte.weight.register_hook(lambda gradient: gradient * float("nan"))
        training = spec.training
        if args.inject_failure == "schedule":
            from dataclasses import replace

            training = replace(
                training, accumulation_steps=training.accumulation_steps + 1
            )
        recovery = None
        started = 0
        if args.resume:
            if args.journal_directory is not None:
                recovery = load_checkpoint_journal(
                    args.journal_directory,
                    model=model,
                    optimizer=optimizer,
                    config=spec.training,
                    source=source,
                )
                started = recovery.step
            else:
                started = load_checkpoint(
                    args.checkpoint,
                    model=model,
                    optimizer=optimizer,
                    config=spec.training,
                    source=source,
                )
        require_all_ranks(
            started + args.steps <= training.optimizer_steps
            and (recovery is None or started + args.steps > recovery.recorded_step)
            and (
                args.failure_after_step is None
                or started < args.failure_after_step <= started + args.steps
            ),
            phase="worker schedule validation",
        )
        if args.observe_recovery:
            _write_boundary(
                args.output.with_suffix(".recovery.json"),
                {
                    "schema": "ai-journey-ddp-restored-boundary-v1",
                    "rank": topology.rank,
                    "world_size": topology.world_size,
                    "pid": os.getpid(),
                    "restored_step": started,
                    "ready_ns": time.monotonic_ns(),
                    "model_sha256": training_model_fingerprint(model),
                },
            )
        metrics = []
        for step in range(started, started + args.steps):
            metrics.extend(
                train_segment(
                    wrapped,
                    optimizer,
                    source,
                    training,
                    topology,
                    start_step=step,
                    steps=1,
                )
            )
            completed = step + 1
            if completed == args.failure_after_step:
                if args.inject_failure == "crash":
                    _abrupt_exit(args.output, topology, completed_step=completed)
                raise RuntimeError(
                    f"injected worker exit after optimizer step {completed}"
                )
            if (
                completed == started + args.steps
                or args.checkpoint_interval is not None
                and completed % args.checkpoint_interval == 0
            ):
                fingerprint = save_checkpoint(
                    args.checkpoint,
                    model=model,
                    optimizer=optimizer,
                    config=spec.training,
                    source=source,
                    step=completed,
                    journal_directory=(
                        args.journal_directory
                        if recovery is None or completed > recovery.recorded_step
                        else None
                    ),
                )
    args.output.write_text(
        json.dumps(
            {
                "rank": args.rank,
                "world_size": args.world_size,
                "pid": __import__("os").getpid(),
                "started_step": started,
                "completed_step": started + args.steps,
                "model_sha256": fingerprint,
                "metrics": metrics,
                **(
                    {
                        "journal_recovery": {
                            "restored_step": recovery.step,
                            "recorded_step": recovery.recorded_step,
                            "checkpoint_sha256": recovery.checkpoint_sha256,
                        }
                    }
                    if recovery is not None
                    else {}
                ),
            },
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
