"""Time one call of each checklist side, then name what bounds each kernel.

``timing`` loads a checklist config, compiles the Day 74 naive source and
the checklist library through ``torch.mps.compile_shader``, times one call
of every side on the three Day 75 clocks at a 1 x 1 x 1 floor and at every
sweep shape, verifies the sealed report, and publishes it atomically without
ever writing over the config. A backend or clock injected through ``main``
yields a report printed and verified as synthetic timing.
``verify-timing`` recomputes a timing report on the CPU, optionally against
the config it must come from.

``bound`` reads a sealed checklist report, a sealed gpu-ceilings report and
a sealed timing report, verifies each with its own verifier, and seals a
bound report with the Day 75 rule config. For each sweep shape and side it
prints the share of the measured FMA ceiling, the modeled traffic, the
attainable rate at its intensity and the Day 75 name, which is an inference
from timings and measured ceilings. For each step kernel it prints the
modeled change beside the measured one. ``verify`` recomputes a bound
report; given the three sources and the rule config, it also rebuilds the
bound from them and requires the same bytes. Neither command runs a kernel
or imports torch.

Exit codes: 0 success; 1 a report was read but failed verification, a
source report included; 2 usage errors, unreadable or invalid inputs,
sources that do not pair, refused outputs, and runs that failed a check, hit
a device error or exceeded their budget; 3 this host cannot run Metal
kernels through PyTorch, so ``timing`` stopped before it measured or wrote
anything. These are the codes of the Day 73 to 77 Metal labs.
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
from .matmul_checklist_bound import SOURCE_NAMES, bound_from_files
from .matmul_checklist_bound_io import (
    BoundSources,
    load_bound_report,
    save_bound_report,
)
from .matmul_checklist_bound_timing import measure_timing_report
from .matmul_checklist_bound_timing_io import load_timing_report, save_timing_report
from .matmul_checklist_bound_traffic import (
    MatmulChecklistBoundError,
    MatmulChecklistBoundVerificationError,
)
from .matmul_checklist_cli import short_name
from .matmul_checklist_config import load_config
from .matmul_checklist_runtime import MatmulChecklistUnavailableError
from .memory_access_report_io import paths_alias

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-matmul-checklist-bound"
CLOCK_FIELDS = (
    ("gpu_ns", "mps_event"),
    ("host_ns", "host_synchronized"),
    ("dispatch_ns", "host_dispatch"),
)
SOURCE_OPTIONS = ("checklist_report", "ceiling_report", "timing_report")
CLAIMS = "cuda=false nvidia=false cublas=false hardware_counters=false"
BOUND_CLAIMS = (
    "counters=false nsight=false instruments=false cuda=false nvidia=false "
    "cublas=false"
)


def _add_sources(parser: argparse.ArgumentParser, *, required: bool) -> None:
    for name in SOURCE_OPTIONS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=required)
    parser.add_argument("--rule-config", type=Path, required=required)


def build_parser() -> argparse.ArgumentParser:
    """Define every command with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Time one call of each Day 77 checklist side on this host's Apple "
            "GPU, and join the checklist rates with this host's measured "
            "ceilings and a traffic model; no CUDA kernel, NVIDIA GPU, cuBLAS or "
            "hardware counter is used."
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
    bound = commands.add_parser(
        "bound",
        help="verify the three source reports and seal the bound of each kernel",
        allow_abbrev=False,
    )
    _add_sources(bound, required=True)
    bound.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify",
        help="recompute a sealed bound report; with the sources, rebuild it",
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


def timing_lines(report: dict[str, object]) -> list[str]:
    """Print the loads, each refusal, each timed shape and the warning count."""

    label = _label(report)
    load = report["host_load"]
    captured = report["warnings"]
    return [
        f"{label} load_average before={_loads(load['before'])} "
        f"after={_loads(load['after'])}",
        *(
            f"{label} refused variant={entry['name']} reason={entry['reason']} "
            f"allowed_threads={entry['allowed_threads']} "
            f"static_threadgroup_bytes={entry['static_threadgroup_bytes']}"
            for entry in report["kernels"]["admissions"]
            if not entry["admitted"]
        ),
        *(
            _one_call_line(label, point)
            for point in (report["floor"], *report["points"])
        ),
        f"{label} warnings count={captured['count']} "
        f"distinct={captured['distinct']} torch_imported_before_capture="
        f"{str(captured['torch_imported_before_capture']).lower()}",
    ]


def timing_summary_line(prefix: str, summary: dict[str, object]) -> str:
    """Print the digest, the counts and the claims of one timing report."""

    counts = (
        f"points={summary['points']} "
        f"sides={','.join(short_name(side) for side in summary['sides'])} "
        f"refused={','.join(summary['refused_variants']) or 'none'} "
        f"warnings={summary['warnings']} "
        f"scope={summary['scope']} "
        f"load_before={_loads(summary['load_average_before'])} "
        f"load_after={_loads(summary['load_average_after'])}"
    )
    if summary["measured"]:
        return f"{prefix}_measurement={summary['report_sha256']} {counts} {CLAIMS}"
    return (
        f"{prefix}_synthetic_timing={summary['report_sha256']} {counts} "
        f"measured=false {CLAIMS}"
    )


def _number(value: float | None, text: str) -> str:
    return "none" if value is None else format(value, text)


def _model(traffic: dict[str, object] | None) -> str:
    if traffic is None:
        return (
            "model=none flops_per_byte=none attainable_gflops=none "
            "attainable_share=none device_gb_per_second=none "
            "threadgroup_reads_per_ns=none"
        )
    roof = traffic["roof"]
    if traffic["block"] is None:
        name = traffic["model"]
    else:
        block = "x".join(str(size) for size in traffic["block"])
        threads = "x".join(str(size) for size in traffic["outputs_per_thread"])
        name = f"tile_{block}_thread_{threads}"
    return (
        f"model={name} "
        f"flops_per_byte={traffic['device']['flops_per_byte']['decimal']} "
        f"attainable_gflops={roof['attainable_gflops_per_second']:.1f} "
        f"attainable_share={roof['attainable_share']:.3f} "
        f"device_gb_per_second={roof['modeled_device_gb_per_second']:.1f} "
        "threadgroup_reads_per_ns="
        f"{_number(roof['modeled_threadgroup_reads_per_ns'], '.1f')}"
    )


def _reads(value: dict[str, object] | None) -> str:
    return "none" if value is None else value["decimal"]


def _step_line(label: str, row: dict[str, object], entry: dict[str, object]) -> str:
    model, measured = entry["model"], entry["measured"]
    reads = model["threadgroup_reads_per_fma"]
    to_base = (
        "none" if measured is None else f"{measured['ratio']:.3f}:{measured['verdict']}"
    )
    held = "none" if measured is None else str(measured["held"]).lower()
    names = entry["bottleneck"]
    return (
        f"{label} step step={row['step']} m={entry['m']} n={entry['n']} "
        f"k={entry['k']} kernel={short_name(row['variant'])} "
        f"base={short_name(row['base'])} "
        f"device_bytes_ratio={model['device_bytes_ratio']['decimal']} "
        f"load_operations_ratio={model['load_operations_ratio']['decimal']} "
        f"threadgroup_reads_per_fma={_reads(reads['base'])}->"
        f"{_reads(reads['variant'])} to_base={to_base} held={held} "
        f'base_name="{names["base"]}" name="{names["variant"]}"'
    )


def bound_lines(report: dict[str, object]) -> list[str]:
    """Print the ceilings, the sources, each side at each point and each step."""

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
                f"{label} bound {shape} side={short_name(side)} "
                f"gflops={entry['median_gflops_per_second']:.1f} "
                f"fma_share={entry['fma_ceiling_share']:.3f} "
                f"{_model(entry['traffic'])} "
                f'name="{decision["bottleneck"]}" reason={decision["reason"]}'
            )
    for row in report["steps"]:
        lines.extend(_step_line(label, row, entry) for entry in row["points"])
    summary = report["summary"]
    for side, top in summary["top_bottleneck"].items():
        lines.append(
            f'{label} top side={short_name(side)} name="{top["top"]}" '
            f"points={top['top_points']}/{summary['points']} reason={top['reason']}"
        )
    return lines


def bound_summary_line(prefix: str, summary: dict[str, object]) -> str:
    """Print the digest, the point count and whether the sources were rebuilt."""

    counts = (
        f"points={summary['points']} "
        f"sources_checked={str(summary['sources_checked']).lower()} "
        f"scope={summary['scope']}"
    )
    if summary["measured"]:
        return (
            f"{prefix}_measurement={summary['report_sha256']} {counts} {BOUND_CLAIMS}"
        )
    return (
        f"{prefix}_synthetic_timing={summary['report_sha256']} {counts} "
        f"measured=false {BOUND_CLAIMS}"
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
        *timing_lines(report),
        timing_summary_line("matmul_checklist_one_call", summary),
    ]


def _bound(args: argparse.Namespace) -> list[str]:
    sources = BoundSources(*(getattr(args, name) for name in SOURCE_OPTIONS))
    protected = (*sources.all(), args.rule_config)
    # Refuse a taken input path before any source is verified.
    if any(paths_alias(args.output, path) for path in protected):
        raise MatmulChecklistBoundError("bound output must not alias an input")
    config = load_rule_config(args.rule_config)
    report = bound_from_files(*sources.all(), config)
    summary = save_bound_report(args.output, report, config=config, protected=protected)
    # bound_from_files verified every source before it built the report.
    checked = {**summary, "sources_checked": True}
    return [
        *bound_lines(report),
        bound_summary_line("matmul_checklist_bound", checked),
    ]


def _verify(args: argparse.Namespace) -> list[str]:
    config = None if args.rule_config is None else load_rule_config(args.rule_config)
    given = [getattr(args, name) for name in SOURCE_OPTIONS]
    sources = None if given[0] is None else BoundSources(*given)
    _, summary = load_bound_report(args.report, config=config, sources=sources)
    return [bound_summary_line("verified_matmul_checklist_bound", summary)]


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
                "verify takes --checklist-report, --ceiling-report and "
                "--timing-report together, or none of them"
            )
        if all(given) and args.rule_config is None:
            parser.error(
                "verify needs --rule-config to rebuild a bound from its sources"
            )
    try:
        if args.command == "verify-timing":
            config = load_config(args.config) if args.config is not None else None
            _, summary = load_timing_report(args.report, config=config)
            lines = [timing_summary_line("verified_matmul_checklist_one_call", summary)]
        elif args.command == "timing":
            lines = _timing(args, backend, clock, load_average)
        elif args.command == "bound":
            lines = _bound(args)
        else:
            lines = _verify(args)
    except MatmulChecklistUnavailableError as error:
        print(
            f"matmul-checklist-bound {args.command} unavailable: {error}",
            file=sys.stderr,
        )
        return EXIT_UNAVAILABLE
    except MatmulChecklistBoundVerificationError as error:
        print(
            f"matmul-checklist-bound {args.command} rejected: {error}", file=sys.stderr
        )
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"matmul-checklist-bound {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in lines:
        print(line)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
