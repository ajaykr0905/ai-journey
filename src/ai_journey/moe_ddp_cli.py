"""Plan, train/resume and inspect bounded public-source replicated CPU experts."""

import argparse
import json
from pathlib import Path

from .ddp_launch import _paths_alias
from .decoder_variant_report_io import write_new_report
from .moe_ddp_evidence import (
    build_moe_ddp_evidence,
    read_moe_ddp_evidence,
    verify_moe_ddp_archive,
)
from .moe_ddp_launch import launch_moe_ddp
from .moe_ddp_plan import read_public_moe_ddp_plan, source_admission
from .moe_ddp_resources import estimate_moe_ddp_job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "train"):
        command = commands.add_parser(name)
        for field in ("config", "source"):
            command.add_argument(f"--{field}", type=Path, required=True)
        if name == "train":
            for field in ("checkpoint", "workdir", "output"):
                command.add_argument(f"--{field}", type=Path, required=True)
            command.add_argument("--steps", type=int, required=True)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--timeout-seconds", type=int, default=60)
    verify = commands.add_parser("verify")
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            report = read_moe_ddp_evidence(args.report)
            result = {
                "evidence_sha256": report["evidence_sha256"],
                "completed_step": report["completed_step"],
                "scope": "static_internal_consistency_not_training_replay",
            }
            if args.checkpoint is not None:
                result["checkpoint"] = verify_moe_ddp_archive(report, args.checkpoint)
        else:
            plan, source = read_public_moe_ddp_plan(args.config, args.source)
            if args.command == "plan":
                result = {
                    "plan_sha256": plan.fingerprint(),
                    "source": source_admission(plan, source),
                    "resources": estimate_moe_ddp_job(plan.run),
                    "execution": "not_executed",
                }
            else:
                lock = args.checkpoint.with_name(
                    f".{args.checkpoint.name}.ai-journey-ddp.lock"
                )
                if (
                    args.output.exists()
                    or args.output.is_symlink()
                    or args.output.resolve().is_relative_to(args.workdir.resolve())
                    or any(
                        _paths_alias(args.output, target)
                        for target in (args.config, args.source, args.checkpoint, lock)
                    )
                ):
                    raise ValueError(
                        "new report must be outside worker outputs and protected inputs"
                    )
                workers = launch_moe_ddp(
                    config=args.config,
                    source=args.source,
                    checkpoint=args.checkpoint,
                    workdir=args.workdir,
                    steps=args.steps,
                    resume=args.resume,
                    timeout_seconds=args.timeout_seconds,
                )
                report = build_moe_ddp_evidence(plan, source, workers)
                verify_moe_ddp_archive(report, args.checkpoint)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_new_report(args.output, report)
                result = {
                    "report": str(args.output),
                    "completed_step": report["completed_step"],
                    "evidence_sha256": report["evidence_sha256"],
                }
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        TimeoutError,
        RecursionError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
