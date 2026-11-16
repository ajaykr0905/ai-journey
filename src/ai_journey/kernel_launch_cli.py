"""Run and verify the simulated kernel-launch lab; no GPU is used.

``run`` loads a plan, simulates every launch in pure Python on the CPU,
seals the report, verifies it and publishes it atomically, never over the
plan. ``verify`` loads a report and recomputes every field from its embedded
plan; ``--config`` also binds the report to a plan file. Both commands print
one summary line.

Exit codes: 0 success; 1 a report was read or built but failed verification;
2 usage errors, unreadable or invalid inputs, and refused outputs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .kernel_launch_config import KernelLaunchError
from .kernel_launch_io import canonical_json_bytes, paths_alias, write_atomic
from .kernel_launch_plan import load_plan
from .kernel_launch_report import DIGEST_FIELD, build_report, seal_report
from .kernel_launch_verify import (
    KernelLaunchVerificationError,
    load_report,
    verify_report,
)

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-journey-kernel-launch",
        description=(
            "Simulated CUDA kernel launches in pure Python on the CPU; no GPU "
            "runs and no CUDA source is compiled."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="simulate a plan and write a sealed report", allow_abbrev=False
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="recompute a sealed report", allow_abbrev=False
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    return parser


def summary_line(prefix: str, report: Mapping[str, object]) -> str:
    """Return the one line that names a verified report and its headline counts."""

    summary = report["summary"]
    return (
        f"{prefix}={report[DIGEST_FIELD]} "
        f"runs={summary['runs']} "
        f"runs_matching_reference={summary['runs_matching_reference']} "
        f"order_independent_kernels={summary['order_independent_kernels']} "
        f"out_of_bounds_writes_caught={summary['out_of_bounds_writes_caught']} "
        f"write_conflicts_caught={summary['write_conflicts_caught']} "
        f"cuda_checks={summary['cuda_checks_passed']}/{summary['cuda_checks']} "
        "simulated_on_cpu=true gpu_executed=false cuda_compiled=false"
    )


def _run(arguments: argparse.Namespace) -> dict[str, object]:
    # Refuse a taken plan path before spending time on the simulation.
    if paths_alias(arguments.output, arguments.config):
        raise KernelLaunchError("report output must not alias the plan")
    plan = load_plan(arguments.config)
    report = seal_report(build_report(plan))
    verify_report(report, plan=plan)
    write_atomic(
        arguments.output,
        canonical_json_bytes(report) + b"\n",
        protected=(arguments.config,),
    )
    return report


def _verify(arguments: argparse.Namespace) -> dict[str, object]:
    plan = None if arguments.config is None else load_plan(arguments.config)
    return load_report(arguments.report, plan=plan)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its explicit exit code."""

    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "run":
            report = _run(arguments)
            prefix = "kernel_launch_report"
        else:
            report = _verify(arguments)
            prefix = "verified_kernel_launch_report"
    except KernelLaunchVerificationError as error:
        print(f"kernel-launch {arguments.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError) as error:
        print(f"kernel-launch {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    print(summary_line(prefix, report))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
