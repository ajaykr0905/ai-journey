"""Analyze, benchmark, and verify the memory-access lab; no GPU is run.

``analyze`` seals an analytical report computed from the declared plan.
``benchmark`` times the plan's float32 NumPy and torch matmuls on this CPU
host and seals a separate measurement report; a clock injected through
``main`` or ``run_benchmark`` yields a report printed and verified as
synthetic timing instead. ``verify`` checks either kind, optionally against
the plan it must have been built from.

Exit codes: 0 success; 1 the report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, and refused outputs.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .memory_access_baseline import cpu_environment, measure_matmul
from .memory_access_plan import MemoryAccessPlan, canonical_json_bytes
from .memory_access_pattern import MemoryAccessError
from .memory_access_report import (
    ANALYSIS_KIND,
    build_analysis_report,
    build_measurement_report,
    seal_report,
)
from .memory_access_report_io import (
    MAX_REPORT_BYTES,
    load_plan,
    parse_json_object,
    paths_alias,
    read_bounded_bytes,
    write_report,
)
from .memory_access_verify import MemoryAccessVerificationError, verify_report

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-journey-memory-access",
        description=(
            "Analytical GPU memory-access models plus real CPU matmul baselines; "
            "no GPU or CUDA kernel is executed."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("analyze", "seal an analytical report computed from the plan"),
        ("benchmark", "measure CPU matmul baselines on this host"),
    ):
        command = commands.add_parser(name, help=help_text, allow_abbrev=False)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="verify an analytical or measurement report", allow_abbrev=False
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    return parser


def run_benchmark(
    plan: MemoryAccessPlan, *, clock: Callable[[], int] = time.perf_counter_ns
) -> dict[str, object]:
    """Measure every planned CPU case and seal the measurement report.

    Any ``clock`` other than the host timer makes the report synthetic.
    """

    benchmark = plan.benchmark
    measurements = tuple(
        measure_matmul(
            case,
            warmup=benchmark.warmup,
            repeats=benchmark.repeats,
            seed=benchmark.seed,
            clock=clock,
        )
        for case in benchmark.cases()
    )
    return seal_report(build_measurement_report(plan, measurements, cpu_environment()))


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    if summary["kind"] == ANALYSIS_KIND:
        return (
            f"{prefix}_analysis={summary['report_sha256']} "
            f"patterns={summary['patterns']} transposes={summary['transposes']} "
            f"tiled_estimates={summary['tiled_estimates']} measured=false "
            "gpu_executed=false"
        )
    if summary["measured"]:
        return (
            f"{prefix}_measurement={summary['report_sha256']} "
            f"cases={summary['cases']} scope={summary['scope']} gpu_executed=false"
        )
    return (
        f"{prefix}_synthetic_timing={summary['report_sha256']} "
        f"cases={summary['cases']} scope={summary['scope']} measured=false "
        "gpu_executed=false"
    )


def _print_measurements(report: dict[str, object]) -> None:
    label = "cpu_host_measurement" if report["measured"] else "synthetic_timing"
    for result in report["results"]:
        print(
            f"{label} "
            f"backend={result['backend']} m={result['m']} n={result['n']} "
            f"k={result['k']} median_ns={result['timing']['median_ns']} "
            f"iqr_ns={result['timing']['iqr_ns']} "
            f"median_gflops_per_second={result['median_gflops_per_second']:.3f}"
        )


def _verify_file(args: argparse.Namespace) -> int:
    plan = load_plan(args.config) if args.config is not None else None
    payload = read_bounded_bytes(
        args.report, max_bytes=MAX_REPORT_BYTES, label="report"
    )
    report = parse_json_object(payload, label="report")
    summary = verify_report(report, plan=plan)
    if payload != canonical_json_bytes(report) + b"\n":
        raise MemoryAccessVerificationError("report file is not canonical JSON")
    print(_summary_line("verified_memory_access", summary))
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None, *, clock: Callable[[], int] | None = None
) -> int:
    """Run one command; ``clock`` replaces the benchmark timer in tests.

    A replaced timer's durations are reported as synthetic, never measured.
    """

    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            return _verify_file(args)
        if paths_alias(args.output, args.config):
            raise MemoryAccessError("report output must not alias the plan")
        plan = load_plan(args.config)
        if args.command == "analyze":
            report = seal_report(build_analysis_report(plan))
        else:
            report = run_benchmark(plan, clock=clock or time.perf_counter_ns)
        summary = verify_report(report, plan=plan)
        write_report(args.output, report, protected=(args.config,))
    except MemoryAccessVerificationError as error:
        print(f"memory-access {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError) as error:
        print(f"memory-access {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    if args.command == "benchmark":
        _print_measurements(report)
    print(_summary_line("memory_access", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
