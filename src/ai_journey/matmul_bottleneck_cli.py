"""Name the top bottleneck of the naive Metal matmul from verified measurements.

``classify`` reads the sealed matmul-metal, gpu-ceilings and roofline gap
reports of Day 74, verifies each with its own verifier, joins the gap with a
one-call timing input, applies the pinned rule and seals the report. It
prints one line per naive kernel and point with the name, its reason and the
numbers behind it, one line per point for the two mappings, and the top name
of each kernel. ``verify`` recomputes a report; given all four inputs, it also
rebuilds the report from them and requires the same bytes. Neither command
runs a kernel, imports torch or reads a hardware counter.

Exit codes: 0 success; 1 a report was read but failed verification, a source
report included; 2 usage errors, unreadable or invalid inputs, inputs that do
not pair, and refused outputs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .matmul_bottleneck_config import (
    MatmulBottleneckError,
    MatmulBottleneckVerificationError,
    load_config,
)
from .matmul_bottleneck_io import (
    InputPaths,
    load_report,
    report_from_files,
    save_report,
)
from .matmul_bottleneck_report import TIMING_SOURCE
from .matmul_roofline_gap import CEILING_SOURCE, NAIVE_SIDES
from .memory_access_report_io import paths_alias

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
PROG = "ai-journey-matmul-bottleneck"
_INPUTS = ("matmul_report", "ceiling_report", "gap_report", "timing")


def _add_inputs(parser: argparse.ArgumentParser, required: bool) -> None:
    for name in _INPUTS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=required)


def build_parser() -> argparse.ArgumentParser:
    """Define the ``classify`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Name the top bottleneck of each naive Metal matmul point from the "
            "verified Day 74 reports and one-call timings; the names are "
            "inferences from timings and measured ceilings, not from hardware "
            "counters."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    classify = commands.add_parser(
        "classify",
        help="verify the Day 74 reports, apply the pinned rule and seal a report",
        allow_abbrev=False,
    )
    _add_inputs(classify, required=True)
    classify.add_argument("--config", type=Path, required=True)
    classify.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify",
        help="recompute a sealed report; with all four inputs, rebuild it",
        allow_abbrev=False,
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    _add_inputs(verify, required=False)
    return parser


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def _lines(report: dict[str, object]) -> list[str]:
    label = _label(report)
    ceilings = report["ceilings"]
    ceiling, timing = (
        report["sources"][CEILING_SOURCE],
        report["sources"][TIMING_SOURCE],
    )
    lines = [
        f"{label} ceilings fma_gflops={ceilings['fma_gflops_per_second']:.1f} "
        f"bandwidth_gb_per_second={ceilings['bandwidth_gb_per_second']:.2f} "
        f"fma_scales_with_work={str(ceilings['fma_scales_with_work']).lower()} "
        f"ceiling_run_load_before={_loads(ceiling['load_average_before'])} "
        f"ceiling_run_load_after={_loads(ceiling['load_average_after'])} "
        f"timing_run_load_before={_loads(timing['load_average_before'])} "
        f"timing_run_load_after={_loads(timing['load_average_after'])}"
    ]
    for point in report["points"]:
        shape = f"m={point['m']} n={point['n']} k={point['k']}"
        for kernel in NAIVE_SIDES:
            entry = point["kernels"][kernel]
            decision = entry["decision"]
            numbers = decision["numbers"]
            lines.append(
                f"{label} bottleneck {shape} mapping={entry['mapping']} "
                f'name="{decision["bottleneck"]}" reason={decision["reason"]} '
                f"overhead_share={numbers['overhead_share']:.3f} "
                f"gpu_to_steady_ratio={numbers['gpu_to_steady_ratio']:.3f} "
                f"fma_share={numbers['fma_ceiling_share']:.3g} "
                f"lower_roof_share={numbers['lower_roof_share']:.3g}"
            )
        mapping = point["mapping"]
        lines.append(
            f"{label} mapping {shape} verdict={mapping['verdict']} "
            f"model_sector_ratio={mapping['model_sector_ratio']:.2f} "
            f"x_columns_to_x_rows_rate_ratio="
            f"{mapping['x_columns_to_x_rows_rate_ratio']:.2f}"
        )
    for kernel, top in report["summary"]["top_bottleneck"].items():
        lines.append(
            f'{label} top kernel={kernel} name="{top["top"]}" '
            f"points={top['top_points']}/{report['summary']['points']} "
            f"reason={top['reason']}"
        )
    return lines


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    counts = (
        f"points={summary['points']} "
        f"sources_checked={str(summary['sources_checked']).lower()} "
        f"scope={summary['scope']}"
    )
    claims = "counters=false nsight=false instruments=false cuda=false nvidia=false"
    if summary["measured"]:
        return f"{prefix}_measurement={summary['report_sha256']} {counts} {claims}"
    return (
        f"{prefix}_synthetic_timing={summary['report_sha256']} {counts} "
        f"measured=false {claims}"
    )


def _inputs(args: argparse.Namespace) -> InputPaths | None:
    given = [getattr(args, name) for name in _INPUTS]
    if all(path is None for path in given):
        return None
    return InputPaths(*given)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "verify":
        given = [getattr(args, name) is not None for name in _INPUTS]
        if any(given) and not all(given):
            parser.error(
                "verify takes --matmul-report, --ceiling-report, --gap-report and "
                "--timing together, or none of them"
            )
    try:
        config = None if args.config is None else load_config(args.config)
        inputs = _inputs(args)
        if args.command == "verify":
            _, summary = load_report(args.report, config=config, inputs=inputs)
            print(_summary_line("verified_matmul_bottleneck", summary))
            return EXIT_OK
        protected = (*inputs.all(), args.config)
        # Refuse a taken input path before any input is verified.
        if any(paths_alias(args.output, path) for path in protected):
            raise MatmulBottleneckError("classify output must not alias an input")
        report = report_from_files(inputs, config)
        summary = save_report(args.output, report, protected=protected)
    except MatmulBottleneckVerificationError as error:
        print(f"matmul-bottleneck {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError) as error:
        print(f"matmul-bottleneck {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in _lines(report):
        print(line)
    # report_from_files verified every source before it built the report.
    print(_summary_line("matmul_bottleneck", {**summary, "sources_checked": True}))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
