"""Time one call of each tiled-lab side, then join the tiled rates with the ceilings.

``timing`` loads a tiled config, compiles the Day 74 naive source and the
tiled library through ``torch.mps.compile_shader``, times one call of every
side on the three Day 75 clocks at a 1 x 1 x 1 floor and at every sweep
shape, verifies the sealed report, and publishes it atomically without ever
writing over the config. A backend or clock injected through ``main`` yields
a report printed and verified as synthetic timing. ``verify-timing``
recomputes a timing report on the CPU, optionally against the config it must
come from.

``gap`` reads a sealed tiled report, a sealed gpu-ceilings report and a
sealed timing report, verifies each with its own verifier, and seals a gap
report with the Day 75 rule config. For each sweep shape and side it prints
the share of the measured FMA ceiling, the modeled traffic, the attainable
rate at its intensity and the Day 75 bottleneck name, which is an inference
from timings and measured ceilings. ``verify`` recomputes a gap report;
given the three sources and the rule config, it also rebuilds the gap from
them and requires the same bytes. Neither command runs a kernel or imports
torch.

Exit codes: 0 success; 1 a report was read but failed verification, a
source report included; 2 usage errors, unreadable or invalid inputs,
sources that do not pair, refused outputs, and runs that failed a check, hit
a device error or exceeded their budget; 3 this host cannot run Metal
kernels through PyTorch, so ``timing`` stopped before it measured or wrote
anything. These are the codes of the Day 73 to 76 Metal labs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .kernel_metal_cli import _refuse_unusable_output
from .matmul_bottleneck_config import load_config as load_rule_config
from .matmul_tiled_cli import short_name
from .matmul_tiled_config import load_config
from .matmul_tiled_gap import SOURCE_NAMES, gap_from_files
from .matmul_tiled_gap_io import GapSources, load_gap_report, save_gap_report
from .matmul_tiled_gap_timing import (
    MatmulTiledGapError,
    MatmulTiledGapVerificationError,
    measure_timing_report,
)
from .matmul_tiled_gap_timing_io import load_timing_report, save_timing_report
from .matmul_tiled_runtime import MatmulTiledUnavailableError
from .memory_access_report_io import paths_alias

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-matmul-tiled-gap"
CLOCK_FIELDS = (
    ("gpu_ns", "mps_event"),
    ("host_ns", "host_synchronized"),
    ("dispatch_ns", "host_dispatch"),
)
SOURCE_OPTIONS = ("tiled_report", "ceiling_report", "timing_report")


def _add_sources(parser: argparse.ArgumentParser, *, required: bool) -> None:
    for name in SOURCE_OPTIONS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=required)
    parser.add_argument("--rule-config", type=Path, required=required)


def build_parser() -> argparse.ArgumentParser:
    """Define every command with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Time one call of each tiled-lab side on this host's Apple GPU, and "
            "join the tiled rates with this host's measured ceilings; no CUDA "
            "kernel, NVIDIA GPU, cuBLAS or hardware counter is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    timing = commands.add_parser(
        "timing",
        help="time one call of every side on three clocks and seal a report",
        allow_abbrev=False,
    )
    timing.add_argument("--config", type=Path, required=True)
    timing.add_argument("--output", type=Path, required=True)
    verify_timing = commands.add_parser(
        "verify-timing",
        help="recompute a sealed timing report on the CPU, without torch",
        allow_abbrev=False,
    )
    verify_timing.add_argument("--report", type=Path, required=True)
    verify_timing.add_argument("--config", type=Path)
    gap = commands.add_parser(
        "gap",
        help="verify the three source reports and seal the gap to the ceilings",
        allow_abbrev=False,
    )
    _add_sources(gap, required=True)
    gap.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify",
        help="recompute a sealed gap report; with the sources, rebuild it",
        allow_abbrev=False,
    )
    verify.add_argument("--report", type=Path, required=True)
    _add_sources(verify, required=False)
    return parser


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def _one_call_line(label: str, point: dict[str, object]) -> str:
    sides = " ".join(
        f"{short_name(side)}_{field}="
        f"{point['sides'][side]['clocks'][clock]['timing']['median_ns']}"
        for side in point["side_order"]
        for field, clock in CLOCK_FIELDS
    )
    return f"{label} one_call m={point['m']} n={point['n']} k={point['k']} {sides}"


def _timing_lines(report: dict[str, object]) -> list[str]:
    label = _label(report)
    load = report["host_load"]
    return [
        f"{label} load_average before={_loads(load['before'])} "
        f"after={_loads(load['after'])}",
        *(
            _one_call_line(label, point)
            for point in (report["floor"], *report["points"])
        ),
    ]


def _timing_summary_line(prefix: str, summary: dict[str, object]) -> str:
    counts = (
        f"points={summary['points']} "
        f"sides={','.join(short_name(side) for side in summary['sides'])} "
        f"scope={summary['scope']} "
        f"load_before={_loads(summary['load_average_before'])} "
        f"load_after={_loads(summary['load_average_after'])}"
    )
    claims = "cuda=false nvidia=false cublas=false"
    if summary["measured"]:
        return f"{prefix}_measurement={summary['report_sha256']} {counts} {claims}"
    return (
        f"{prefix}_synthetic_timing={summary['report_sha256']} {counts} "
        f"measured=false {claims}"
    )


def _model(traffic: dict[str, object] | None) -> str:
    if traffic is None:
        return (
            "model=none model_flops_per_byte=none attainable_gflops=none "
            "attainable_share=none"
        )
    name = traffic["model"] if traffic["tile"] is None else f"tiled_{traffic['tile']}"
    return (
        f"model={name} "
        f"model_flops_per_byte={traffic['flops_per_byte']['decimal']} "
        f"attainable_gflops={traffic['attainable_gflops_per_second']:.1f} "
        f"attainable_share={traffic['attainable_share']:.3g}"
    )


def _gap_lines(report: dict[str, object]) -> list[str]:
    label = _label(report)
    ceilings = report["ceilings"]
    lines = [
        f"{label} ceilings fma_gflops={ceilings['fma_gflops_per_second']:.1f} "
        f"bandwidth_gb_per_second={ceilings['bandwidth_gb_per_second']:.2f} "
        f"ridge_flops_per_byte={ceilings['ridge_flops_per_byte']:.2f} "
        f"fma_scales_with_work={str(ceilings['fma_scales_with_work']).lower()}"
    ]
    for name in SOURCE_NAMES:
        source = report["sources"][name]
        lines.append(
            f"{label} source name={name} report={source['report_sha256']} "
            f"load_before={_loads(source['load_average_before'])} "
            f"load_after={_loads(source['load_average_after'])}"
        )
    for point in report["points"]:
        shape = f"m={point['m']} n={point['n']} k={point['k']}"
        ideal = point["ideal_reuse"]
        lines.append(
            f"{label} roof {shape} "
            f"ideal_flops_per_byte={ideal['flops_per_byte']['decimal']} "
            f"ideal_roof_gflops={ideal['bandwidth_roof_gflops_per_second']:.1f} "
            f"ideal_bound={ideal['bound']}"
        )
        for side in report["sides"]:
            entry = point["sides"][side]
            decision = entry["decision"]
            lines.append(
                f"{label} gap {shape} side={short_name(side)} "
                f"gflops={entry['median_gflops_per_second']:.1f} "
                f"fma_share={entry['fma_ceiling_share']:.3g} "
                f"{_model(entry['traffic'])} "
                f"ideal_share={entry['ideal_reuse_share']:.3g} "
                f'name="{decision["bottleneck"]}" reason={decision["reason"]}'
            )
    summary = report["summary"]
    for side, top in summary["top_bottleneck"].items():
        lines.append(
            f'{label} top side={short_name(side)} name="{top["top"]}" '
            f"points={top['top_points']}/{summary['points']} reason={top['reason']}"
        )
    return lines


def _gap_summary_line(prefix: str, summary: dict[str, object]) -> str:
    counts = (
        f"points={summary['points']} "
        f"sources_checked={str(summary['sources_checked']).lower()} "
        f"scope={summary['scope']}"
    )
    claims = (
        "counters=false nsight=false instruments=false cuda=false nvidia=false "
        "cublas=false"
    )
    if summary["measured"]:
        return f"{prefix}_measurement={summary['report_sha256']} {counts} {claims}"
    return (
        f"{prefix}_synthetic_timing={summary['report_sha256']} {counts} "
        f"measured=false {claims}"
    )


def _timing(
    args: argparse.Namespace,
    backend: object | None,
    clock: Callable[[], int] | None,
    load_average: Callable[[], tuple[float, float, float]] | None,
) -> list[str]:
    _refuse_unusable_output(args.output, args.config)
    config = load_config(args.config)
    report = measure_timing_report(
        config,
        backend=backend,
        clock=clock or time.perf_counter_ns,
        load_average=load_average or os.getloadavg,
    )
    summary = save_timing_report(
        args.output, report, config=config, protected=(args.config,)
    )
    return [
        *_timing_lines(report),
        _timing_summary_line("matmul_tiled_one_call", summary),
    ]


def _gap(args: argparse.Namespace) -> list[str]:
    sources = GapSources(*(getattr(args, name) for name in SOURCE_OPTIONS))
    protected = (*sources.all(), args.rule_config)
    # Refuse a taken input path before any source is verified.
    if any(paths_alias(args.output, path) for path in protected):
        raise MatmulTiledGapError("gap output must not alias an input")
    config = load_rule_config(args.rule_config)
    report = gap_from_files(*sources.all(), config)
    summary = save_gap_report(args.output, report, config=config, protected=protected)
    # gap_from_files verified every source before it built the report.
    checked = {**summary, "sources_checked": True}
    return [*_gap_lines(report), _gap_summary_line("matmul_tiled_gap", checked)]


def _verify(args: argparse.Namespace) -> list[str]:
    config = None if args.rule_config is None else load_rule_config(args.rule_config)
    given = [getattr(args, name) for name in SOURCE_OPTIONS]
    sources = None if given[0] is None else GapSources(*given)
    _, summary = load_gap_report(args.report, config=config, sources=sources)
    return [_gap_summary_line("verified_matmul_tiled_gap", summary)]


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: object | None = None,
    clock: Callable[[], int] | None = None,
    load_average: Callable[[], tuple[float, float, float]] | None = None,
) -> int:
    """Run one command and return its exit code; the keywords are for tests.

    A replaced backend or clock makes a timing report synthetic, never a
    measurement.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "verify":
        given = [getattr(args, name) is not None for name in SOURCE_OPTIONS]
        if any(given) and not all(given):
            parser.error(
                "verify takes --tiled-report, --ceiling-report and --timing-report "
                "together, or none of them"
            )
        if all(given) and args.rule_config is None:
            parser.error("verify needs --rule-config to rebuild a gap from its sources")
    try:
        if args.command == "verify-timing":
            config = load_config(args.config) if args.config is not None else None
            _, summary = load_timing_report(args.report, config=config)
            lines = [_timing_summary_line("verified_matmul_tiled_one_call", summary)]
        elif args.command == "timing":
            lines = _timing(args, backend, clock, load_average)
        elif args.command == "gap":
            lines = _gap(args)
        else:
            lines = _verify(args)
    except MatmulTiledUnavailableError as error:
        print(f"matmul-tiled-gap {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except MatmulTiledGapVerificationError as error:
        print(f"matmul-tiled-gap {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"matmul-tiled-gap {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in lines:
        print(line)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
