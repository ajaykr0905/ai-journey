"""Build and verify analytical SIMT execution-model reports; no GPU is run.

``build`` loads a declared plan, derives every scenario, seals the report and
publishes it atomically without ever writing over the plan. ``verify`` loads a
report and independently recomputes every field. Both print one summary line
and return 0; invalid plans, reports, or paths return 1, and argparse usage
errors exit with 2.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from .simt_device import SimtError
from .simt_plan import load_simt_plan
from .simt_report import build_simt_report, seal_simt_report
from .simt_report_io import load_simt_report, save_simt_report

EXIT_OK = 0
EXIT_INVALID = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-journey-simt",
        description=(
            "Analytical SIMT execution-model reports from declared inputs; "
            "no GPU or CUDA kernel is run."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="write a sealed report from a plan", allow_abbrev=False
    )
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="independently recompute a report", allow_abbrev=False
    )
    verify.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code."""

    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "build":
            plan = load_simt_plan(arguments.config)
            report = seal_simt_report(build_simt_report(plan))
            save_simt_report(arguments.output, report, protected=(arguments.config,))
            label = "simt_report"
        else:
            report = load_simt_report(arguments.report)
            label = "verified_simt_report"
    except (SimtError, OSError) as error:
        print(f"ai-journey-simt {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    summary = report["summary"]
    print(
        f"{label}={report['integrity']['digest']} "
        f"devices={summary['devices']} "
        f"sweep_cells={summary['occupancy_sweep_cells']} "
        f"memory_rows={summary['memory_latency_rows']} "
        f"arithmetic_rows={summary['arithmetic_latency_rows']} "
        "measured=false gpu_executed=false"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
