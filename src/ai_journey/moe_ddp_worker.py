"""One owned CPU/Gloo replica for the strict public-byte expert protocol."""

import argparse
from dataclasses import asdict
import os
from pathlib import Path

import torch

from .ddp_contracts import Topology
from .ddp_runtime import process_group, require_all_ranks
from .decoder_variant_report_io import write_new_report
from .moe_ddp_checkpoint import load_moe_ddp_checkpoint, save_moe_ddp_checkpoint
from .moe_ddp_data import public_moe_ddp_window
from .moe_ddp_plan import read_public_moe_ddp_plan
from .moe_ddp_training import MoEDDPTrainingSession
from .training_models import training_model_fingerprint

RECEIPT_SCHEMA = "ai-journey-public-moe-ddp-rank-v1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rendezvous", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    plan, source = read_public_moe_ddp_plan(args.config, args.source)
    if not 1 <= args.steps <= plan.run.optimizer_steps:
        raise ValueError("worker steps exceed the admitted full schedule")
    topology = Topology(args.rank, plan.run.world_size)
    torch.set_num_threads(1)
    with process_group(topology, rendezvous=args.rendezvous):
        session = MoEDDPTrainingSession(plan.run)
        torch.set_rng_state(
            torch.Generator(device="cpu")
            .manual_seed(plan.run.seed + 1000 + args.rank)
            .get_state()
        )
        started = (
            load_moe_ddp_checkpoint(
                args.checkpoint,
                session=session,
                source=source,
                data_contract=plan.data_contract(),
            )
            if args.resume
            else 0
        )
        require_all_ranks(
            started + args.steps <= plan.run.optimizer_steps,
            phase="public MoE resumed schedule admission",
        )
        metrics = []
        for index in range(started, started + args.steps):
            window, schedule = public_moe_ddp_window(
                plan, source, rank=args.rank, update_index=index
            )
            metric = session.step(window)
            metric["data_schedule"] = schedule
            metrics.append(metric)
        identity = save_moe_ddp_checkpoint(
            args.checkpoint,
            session=session,
            source=source,
            data_contract=plan.data_contract(),
        )
        write_new_report(
            args.output,
            {
                "schema": RECEIPT_SCHEMA,
                "rank": args.rank,
                "world_size": plan.run.world_size,
                "pid": os.getpid(),
                "plan_sha256": plan.fingerprint(),
                "started_step": started,
                "completed_step": started + args.steps,
                "model_sha256": training_model_fingerprint(session.model),
                "checkpoint_identity": asdict(identity),
                "metrics": metrics,
            },
        )


if __name__ == "__main__":
    main()
