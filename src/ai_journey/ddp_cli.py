"""Train, verify and certify the bounded CPU distributed GPT-2 lab."""

import argparse
import json
from pathlib import Path

from .ddp_acceptance import (
    certify_recovery,
    certify_batch_equivalence,
    certify_crash_recovery,
    read_crash_recovery,
)
from .ddp_evidence import build_evidence, read_evidence, write_evidence
from .ddp_launch import launch_training
from .gpt2_training import load_training_spec
from .ddp_operations import (
    _validate_operation_paths,
    add_commands,
    run_operation,
    write_report,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "recover", "compare"):
        command = commands.add_parser(name)
        for field in ("source", "spec", "output", "workdir"):
            command.add_argument(f"--{field}", type=Path, required=True)
        command.add_argument("--world-size", type=int, default=2)
        command.add_argument("--steps", type=int, default=2)
        if name == "train":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--checkpoint-interval", type=int)
    verify = commands.add_parser("verify")
    verify.add_argument("--report", type=Path, required=True)
    crash = commands.add_parser("recover-crash")
    for field in ("source", "spec", "output", "workdir"):
        crash.add_argument(f"--{field}", type=Path, required=True)
    crash.add_argument("--world-size", type=int, default=2)
    crash.add_argument("--steps", type=int, default=4)
    crash.add_argument("--checkpoint-interval", type=int, default=2)
    crash.add_argument("--failure-after-step", type=int, default=3)
    crash.add_argument("--failure-rank", type=int, default=1)
    verify_crash = commands.add_parser("verify-crash-recovery")
    verify_crash.add_argument("--report", type=Path, required=True)
    verify_crash.add_argument("--workdir", type=Path)
    add_commands(commands)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command in ("train", "recover", "compare", "recover-crash"):
            _validate_operation_paths(arguments)
        if arguments.command in (
            "matrix",
            "pipeline",
            "serve",
            "benchmark",
            "verify-comparison",
        ):
            result = run_operation(arguments)
        elif arguments.command == "verify":
            result = read_evidence(arguments.report)
        elif arguments.command == "verify-crash-recovery":
            result = read_crash_recovery(arguments.report, directory=arguments.workdir)
        elif arguments.command == "recover-crash":
            result = certify_crash_recovery(
                source=arguments.source,
                spec=arguments.spec,
                directory=arguments.workdir,
                world_size=arguments.world_size,
                total_steps=arguments.steps,
                checkpoint_interval=arguments.checkpoint_interval,
                failure_after_step=arguments.failure_after_step,
                failure_rank=arguments.failure_rank,
            )
            write_report(arguments.output, result)
        elif arguments.command == "train":
            spec = load_training_spec(arguments.spec)
            ranks = launch_training(
                source=arguments.source,
                spec=arguments.spec,
                checkpoint=arguments.checkpoint,
                output_dir=arguments.workdir,
                world_size=arguments.world_size,
                steps=arguments.steps,
                resume=arguments.resume,
                checkpoint_interval=arguments.checkpoint_interval,
            )
            result = build_evidence(arguments.source.read_bytes(), spec, ranks)
            write_evidence(arguments.output, result)
        else:
            function = (
                certify_recovery
                if arguments.command == "recover"
                else certify_batch_equivalence
            )
            step_field = "split_steps" if arguments.command == "recover" else "steps"
            result = function(
                source=arguments.source,
                spec=arguments.spec,
                directory=arguments.workdir,
                world_size=arguments.world_size,
                **{step_field: arguments.steps},
            )
            write_report(arguments.output, result)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
