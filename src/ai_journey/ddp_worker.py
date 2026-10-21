"""One bounded CPU/Gloo worker, launched only by the local lab supervisor."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from .ddp_checkpoint import save_checkpoint, load_checkpoint
from .ddp_contracts import Topology
from .ddp_runtime import process_group
from .ddp_training import train_segment
from .gpt2_model import GPT2LanguageModel
from .gpt2_training import load_training_spec, build_gpt2_optimizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "spec", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rendezvous", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inject-failure", choices=("exit", "stall"))
    args = parser.parse_args(argv)
    if args.inject_failure == "exit":
        raise RuntimeError("injected worker exit before process-group initialization")
    if args.inject_failure == "stall":
        import time

        time.sleep(30)
    topology = Topology(args.rank, args.world_size, args.rank)
    spec = load_training_spec(args.spec)
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
        metrics = train_segment(
            wrapped,
            optimizer,
            source,
            spec.training,
            topology,
            start_step=started,
            steps=args.steps,
        )
        fingerprint = save_checkpoint(
            args.checkpoint,
            model=model,
            optimizer=optimizer,
            config=spec.training,
            source=source,
            step=started + args.steps,
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
