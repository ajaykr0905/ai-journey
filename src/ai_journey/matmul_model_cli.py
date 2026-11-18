"""Run and verify the matmul model lab; no GPU is used.

``run`` loads a plan, simulates and models every case in pure Python on the
CPU, seals the report, verifies it and publishes it atomically, never over
the plan. ``verify`` loads a report and recomputes every field from its
embedded plan; ``--config`` also binds the report to a plan file. Both
commands print one summary line.

Exit codes: 0 success; 1 a report was read or built but failed verification;
2 usage errors, unreadable or invalid inputs, and refused outputs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .kernel_launch_io import canonical_json_bytes, paths_alias, write_atomic
from .matmul_model_plan import load_plan
from .matmul_model_reference import MatmulModelError
from .matmul_model_report import DIGEST_FIELD, build_report, seal_report
from .matmul_model_verify import (
    MatmulModelVerificationError,
    load_report,
    verify_report,
)

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
PROG = "ai-journey-matmul-model"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Naive matmul model in pure Python on the CPU: simulated launches, "
            "warp access counts, FLOP and byte accounting, a roofline at "
            "declared ceilings and an occupancy estimate. No GPU runs, and no "
            "CUDA source is compiled."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="model a plan and write a sealed report", allow_abbrev=False
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
        "order_independent_launches="
        f"{summary['order_independent_launches']}/{summary['launches']} "
        f"cross_checked_requests={summary['cross_checked_requests']} "
        f"probe_traces={summary['probe_traces_matching_model']}/{summary['probes']} "
        "memory_bound_points="
        f"{summary['memory_bound_points']}/{summary['roofline_points']} "
        f"cuda_checks={summary['cuda_checks_passed']}/{summary['cuda_checks']} "
        "simulated_on_cpu=true gpu_executed=false cuda_compiled=false"
    )


def _run(arguments: argparse.Namespace) -> dict[str, object]:
    # Refuse a taken plan path before spending time on the simulation.
    if paths_alias(arguments.output, arguments.config):
        raise MatmulModelError("report output must not alias the plan")
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
            prefix = "matmul_model_report"
        else:
            report = _verify(arguments)
            prefix = "verified_matmul_model_report"
    except MatmulModelVerificationError as error:
        print(f"matmul-model {arguments.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError) as error:
        print(f"matmul-model {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    print(summary_line(prefix, report))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
