"""Compare naive Metal matmul rates with this host's measured ceilings.

``gap`` reads a sealed matmul-metal report and a sealed gpu-ceilings report,
verifies each with its own verifier, and seals a gap report. For each sweep
shape and mapping it prints the share of the measured FMA ceiling, the share
of the bandwidth roof at the model lab's no-reuse and ideal-reuse
intensities, and the ratio to ``torch.matmul``. ``verify`` recomputes a gap
report; given both source reports, it also rebuilds the gap from them and
requires the same bytes. Neither command runs a kernel or imports torch.

Exit codes: 0 success; 1 a report was read but failed verification, a source
report included; 2 usage errors, unreadable or invalid inputs, sources that
do not pair, and refused outputs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .matmul_metal_runtime import TORCH_MATMUL
from .matmul_model_traffic import TRAFFIC_MODELS
from .matmul_roofline_gap import (
    CEILING_SOURCE,
    NAIVE_SIDES,
    MatmulRooflineError,
    MatmulRooflineVerificationError,
    gap_from_files,
)
from .matmul_roofline_io import load_report, save_report
from .memory_access_report_io import paths_alias

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
PROG = "ai-journey-matmul-roofline"


def build_parser() -> argparse.ArgumentParser:
    """Define the ``gap`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Compare the naive Metal matmul rates of a sealed matmul-metal report "
            "with the measured ceilings of a sealed gpu-ceilings report from the "
            "same host; no kernel runs, and no vendor peak is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    gap = commands.add_parser(
        "gap",
        help="verify both source reports and seal the gap to the ceilings",
        allow_abbrev=False,
    )
    gap.add_argument("--matmul-report", type=Path, required=True)
    gap.add_argument("--ceiling-report", type=Path, required=True)
    gap.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify",
        help="recompute a sealed gap report; with both sources, rebuild it",
        allow_abbrev=False,
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--matmul-report", type=Path)
    verify.add_argument("--ceiling-report", type=Path)
    return parser


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def _lines(report: dict[str, object]) -> list[str]:
    label = _label(report)
    ceilings = report["ceilings"]
    source = report["sources"][CEILING_SOURCE]
    lines = [
        f"{label} ceilings fma_gflops={ceilings['fma_gflops_per_second']:.1f} "
        f"bandwidth_gb_per_second={ceilings['bandwidth_gb_per_second']:.2f} "
        f"ridge_flops_per_byte={ceilings['ridge_flops_per_byte']:.2f} "
        f"fma_scales_with_work={str(ceilings['fma_scales_with_work']).lower()} "
        f"ceiling_run_load_before={_loads(source['load_average_before'])} "
        f"ceiling_run_load_after={_loads(source['load_average_after'])}"
    ]
    for point in report["points"]:
        shape = f"m={point['m']} n={point['n']} k={point['k']}"
        roofs = " ".join(
            f"{model}_flops_per_byte="
            f"{point['intensities'][model]['flops_per_byte']['decimal']} "
            f"{model}_roof_gflops="
            f"{point['intensities'][model]['bandwidth_roof_gflops_per_second']:.1f} "
            f"{model}_bound={point['intensities'][model]['bound']}"
            for model in TRAFFIC_MODELS
        )
        torch = point["sides"][TORCH_MATMUL]
        lines.append(
            f"{label} roof {shape} {roofs} "
            f"torch_gflops={torch['median_gflops_per_second']:.1f} "
            f"torch_fma_share={torch['fma_ceiling_share']:.3g}"
        )
        for side in NAIVE_SIDES:
            entry = point["sides"][side]
            lines.append(
                f"{label} gap {shape} mapping={entry['mapping']} "
                f"gflops={entry['median_gflops_per_second']:.1f} "
                f"fma_share={entry['fma_ceiling_share']:.3g} "
                f"no_reuse_roof_share={entry['no_reuse_roof_share']:.3g} "
                f"ideal_reuse_roof_share={entry['ideal_reuse_roof_share']:.3g} "
                f"torch_ratio={entry['torch_ratio']:.3f}"
            )
    return lines


def _summary_line(prefix: str, summary: dict[str, object], sources: bool) -> str:
    counts = (
        f"points={summary['points']} sources_checked={str(sources).lower()} "
        f"scope={summary['scope']}"
    )
    if summary["measured"]:
        return (
            f"{prefix}_measurement={summary['report_sha256']} {counts} "
            "cuda=false nvidia=false"
        )
    return (
        f"{prefix}_synthetic_timing={summary['report_sha256']} {counts} "
        "measured=false cuda=false nvidia=false"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "verify" and (args.matmul_report is None) != (
        args.ceiling_report is None
    ):
        parser.error(
            "verify takes both --matmul-report and --ceiling-report, or neither"
        )
    try:
        if args.command == "verify":
            _, summary = load_report(
                args.report,
                matmul_report=args.matmul_report,
                ceiling_report=args.ceiling_report,
            )
            print(
                _summary_line(
                    "verified_matmul_roofline", summary, summary["sources_checked"]
                )
            )
            return EXIT_OK
        # Refuse a taken source path before the sources are verified.
        for source in (args.matmul_report, args.ceiling_report):
            if paths_alias(args.output, source):
                raise MatmulRooflineError("gap output must not alias a source report")
        report = gap_from_files(args.matmul_report, args.ceiling_report)
        summary = save_report(
            args.output, report, protected=(args.matmul_report, args.ceiling_report)
        )
    except MatmulRooflineVerificationError as error:
        print(f"matmul-roofline {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError) as error:
        print(f"matmul-roofline {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in _lines(report):
        print(line)
    # gap_from_files verified both sources before it built the report.
    print(_summary_line("matmul_roofline", summary, True))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
