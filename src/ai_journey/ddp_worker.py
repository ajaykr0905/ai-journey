"""One bounded CPU/Gloo worker, launched only by the local lab supervisor."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from .ddp_checkpoint import save_checkpoint, load_checkpoint
from .ddp_contracts import Topology, integer
from .ddp_runtime import process_group, require_all_ranks
from .ddp_training import train_segment
from .gpt2_model import GPT2LanguageModel
from .gpt2_training import build_gpt2_optimizer
from .ddp_limits import preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "spec", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rendezvous", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--failure-after-step", type=int)
    parser.add_argument(
        "--inject-failure", choices=("exit", "stall", "nonfinite", "schedule")
    )
    args = parser.parse_args(argv)
    if args.checkpoint_interval is not None:
        integer(args.checkpoint_interval, "checkpoint_interval", 1)
    if args.failure_after_step is not None:
        integer(args.failure_after_step, "failure_after_step", 1)
        if args.inject_failure != "exit":
            raise ValueError("late failure supports exit mode only")
    if args.inject_failure == "exit" and args.failure_after_step is None:
        raise RuntimeError("injected worker exit before process-group initialization")
    if args.inject_failure == "stall":
        import time

        time.sleep(30)
    topology = Topology(args.rank, args.world_size, args.rank)
    spec = preflight(
        args.source, args.spec, world_size=args.world_size, steps=args.steps
    )
    source = args.source.read_bytes()
    from hashlib import sha256

    if sha256(source).hexdigest() != spec.source.sha256:
        raise ValueError("public source digest does not match specification")
    torch.set_num_threads(1)
    torch.manual_seed(spec.training.seed)
    model = GPT2LanguageModel(spec.model).double()
    optimizer = build_gpt2_optimizer(model, spec.training)
    with process_group(topology, rendezvous=args.rendezvous):
        wrapped = DistributedDataParallel(model)
        if args.inject_failure == "nonfinite":
            model.wte.weight.register_hook(lambda gradient: gradient * float("nan"))
        training = spec.training
        if args.inject_failure == "schedule":
            from dataclasses import replace

            training = replace(
                training, accumulation_steps=training.accumulation_steps + 1
            )
        started = (
            load_checkpoint(
                args.checkpoint,
                model=model,
                optimizer=optimizer,
                config=spec.training,
                source=source,
            )
            if args.resume
            else 0
        )
        require_all_ranks(
            started + args.steps <= training.optimizer_steps
            and (
                args.failure_after_step is None
                or started < args.failure_after_step <= started + args.steps
            ),
            phase="worker schedule validation",
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
            },
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
