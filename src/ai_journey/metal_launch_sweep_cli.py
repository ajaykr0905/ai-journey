"""Run and verify the Day 74 launch-shape sweep on this host's Apple GPU.

``run`` loads a config, compiles the sweep kernels beside the Day 73 ones,
admits every shape against its pipeline, checks the mapping, the SIMD layout
and every correctness case, times every sweep point, verifies the sealed
report, and publishes it atomically without ever writing over the config. A
backend or clock injected through ``main`` yields a report printed and
verified as synthetic timing. ``verify`` recomputes a report on the CPU,
optionally against the config it must come from, and never imports torch.

Exit codes are those of the Day 73 Metal lab: 0 success; 1 a report was read
but failed verification; 2 usage errors, unreadable or invalid inputs,
refused outputs, and runs that failed a check, hit a device error or
exceeded their budget; 3 this host cannot run Metal kernels through PyTorch
(not macOS, no torch, no MPS device or no ``torch.mps.compile_shader``), so
``run`` stopped before it measured anything. PyTorch raises ``RuntimeError``
when the MPS device fails, for example when it runs out of memory; that is
exit 2, so exit 1 keeps its one meaning.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .kernel_metal_cli import _refuse_unusable_output
from .kernel_metal_runtime import KernelMetalUnavailableError
from .metal_launch_sweep_config import load_config
from .metal_launch_sweep_io import load_report, save_report
from .metal_launch_sweep_report import measure_report
from .metal_launch_sweep_verify import MetalLaunchSweepVerificationError

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-metal-launch-sweep"


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Sweep the threadgroup shapes of a naive Metal matmul and the group "
            "sizes of a vector add on this host's Apple GPU; no CUDA kernel or "
            "NVIDIA GPU is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="check and time every launch shape on this Apple GPU and seal a report",
        allow_abbrev=False,
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify",
        help="recompute a sealed report on the CPU, without torch",
        allow_abbrev=False,
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    return parser


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _shape(group: list[int]) -> str:
    return "x".join(str(size) for size in group)


def _loads(values: object) -> str:
    if values is None:
        return "unknown"
    return ",".join(f"{value:.2f}" for value in values)


def _point_lines(label: str, sweep: str, point: dict[str, object]) -> list[str]:
    if sweep == "matmul":
        extent = f"m={point['m']} n={point['n']} k={point['k']}"
        rate, digits = "gflops_per_second", 1
    else:
        extent = f"length={point['length']}"
        rate, digits = "gb_per_second", 2
    lines = [
        f"{label} {sweep} {extent} group={_shape(shape['group_size'])} "
        f"rank={shape['rank']} median_us={shape['median_ns_per_call'] / 1000:.2f} "
        f"iqr_us={shape['iqr_ns_per_call'] / 1000:.2f} "
        f"{rate}={shape[f'median_{rate}']:.{digits}f} "
        f"relative_to_fastest={shape['relative_to_fastest']:.3f}"
        for shape in point["shapes"]
    ]
    order = ",".join(_shape(group) for group in point["ranking"])
    lines.append(
        f"{label} ranking {sweep} {extent} order={order} "
        f"spread={point['spread']:.3f} separated={str(point['separated']).lower()}"
    )
    return lines


def _print_run(report: dict[str, object]) -> None:
    label = _label(report)
    rule = str(report["dispatch"]["requires_whole_simd_groups"]).lower()
    for name, limits in sorted(report["pipelines"].items()):
        print(
            f"{label} pipeline kernel={name} "
            f"max_threads_per_threadgroup={limits['max_threads_per_threadgroup']} "
            f"thread_execution_width={limits['thread_execution_width']} "
            f"requires_whole_simd_groups={rule}"
        )
    for sweep in ("matmul", "vector"):
        for entry in report["admission"][sweep]:
            verdict = (
                "admitted=true"
                if entry["admitted"]
                else f"admitted=false reason={entry['reason']}"
            )
            print(
                f"{label} admission sweep={sweep} group={_shape(entry['group_size'])} "
                f"threads={entry['threads_per_group']} "
                f"simd_groups={entry['simd_groups']} idle_lanes={entry['idle_lanes']} "
                f"rows_per_simd_group={entry['rows_per_simd_group']} "
                f"columns_per_simd_group={entry['columns_per_simd_group']} {verdict}"
            )
    summary = report["summary"]
    correctness = report["correctness"]
    print(
        f"{label} correctness matmul cases={len(correctness['matmul'])} "
        f"mapping_checks={summary['mapping_checks']} "
        f"layout_probes={summary['layout_probes']} "
        f"max_bound_ratio={summary['matmul_max_bound_ratio']:.4f}"
    )
    print(
        f"{label} correctness vector_add cases={len(correctness['vector_add'])} "
        "implementations=metal_kernel/mps,torch/mps,torch/cpu"
    )
    for sweep in ("matmul", "vector"):
        for point in report["benchmark"][sweep]:
            for line in _point_lines(label, sweep, point):
                print(line)
    run = report["run"]
    print(
        f"{label} load_average before={_loads(run['load_average_before'])} "
        f"after={_loads(run['load_average_after'])}"
    )


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    counts = (
        f"admitted_shapes={summary['admitted_shapes']} "
        f"rejected_shapes={summary['rejected_shapes']} "
        f"correctness_cases={summary['correctness_cases']} "
        f"benchmark_points={summary['benchmark_points']} scope={summary['scope']}"
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


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: object | None = None,
    clock: Callable[[], int] | None = None,
) -> int:
    """Run one command and return its exit code; ``backend`` and ``clock`` are for tests.

    A replaced backend or clock makes the report synthetic, never a
    measurement.
    """

    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            config = load_config(args.config) if args.config is not None else None
            _, summary = load_report(args.report, config=config)
            print(_summary_line("verified_metal_launch_sweep", summary))
            return EXIT_OK
        _refuse_unusable_output(args.output, args.config)
        config = load_config(args.config)
        report = measure_report(
            config, backend=backend, clock=clock or time.perf_counter_ns
        )
        summary = save_report(
            args.output, report, config=config, protected=(args.config,)
        )
    except KernelMetalUnavailableError as error:
        print(
            f"metal-launch-sweep {args.command} unavailable: {error}", file=sys.stderr
        )
        return EXIT_UNAVAILABLE
    except MetalLaunchSweepVerificationError as error:
        print(f"metal-launch-sweep {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"metal-launch-sweep {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    _print_run(report)
    print(_summary_line("metal_launch_sweep", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
