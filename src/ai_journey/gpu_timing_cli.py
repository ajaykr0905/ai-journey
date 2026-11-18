"""Run and verify the three-clock GPU timing lab on this host's Apple GPU.

``run`` loads a config, compiles the Day 74 matmul kernels and the empty
kernel through ``torch.mps.compile_shader``, times every side on the MPS
event clock and on two host clocks, verifies the sealed report, and
publishes it atomically without ever writing over the config. A backend or
clock injected through ``main`` yields a report printed and verified as
synthetic timing. ``verify`` recomputes a report on the CPU, optionally
against the config it must come from, and never imports torch.

Exit codes: 0 success; 1 a report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, refused outputs, and runs that failed
a check, hit a device error or exceeded their budget; 3 this host cannot run
Metal kernels through PyTorch (not macOS, no torch, no MPS device or no
``torch.mps.compile_shader``), so ``run`` stopped before it measured
anything. These are the Day 73 and 74 codes, so scripts treat all three
Metal labs alike.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .gpu_timing_config import load_config
from .gpu_timing_io import load_report, save_report
from .gpu_timing_report import measure_report
from .gpu_timing_runtime import GpuTimingUnavailableError
from .gpu_timing_verify import GpuTimingVerificationError
from .kernel_metal_cli import _refuse_unusable_output

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-gpu-timing"
SHORT_NAMES = {
    "matmul_x_columns": "x_columns",
    "matmul_x_rows": "x_rows",
    "torch_matmul": "torch",
}
SHORT_CLOCKS = {
    "mps_event": "event",
    "host_synchronized": "sync",
    "host_dispatch": "dispatch",
}


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Time the Day 74 naive Metal matmuls and torch.matmul on three clocks "
            "(MPS events, synchronized host time, dispatch host time) on this "
            "host's Apple GPU; no CUDA kernel or NVIDIA GPU is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="time the kernels on this Apple GPU and seal a report",
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


def _us(value: float) -> str:
    return f"{value / 1e3:.1f}"


def _clocks(clocks: dict[str, object]) -> str:
    return " ".join(
        f"{short}_median_us={_us(clocks[clock]['median_ns'])} "
        f"{short}_iqr_us={_us(clocks[clock]['iqr_ns'])}"
        for clock, short in SHORT_CLOCKS.items()
    )


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def _print_run(report: dict[str, object]) -> None:
    label = _label(report)
    analysis = report["analysis"]
    load = report["host_load"]
    print(
        f"{label} load_average before={_loads(load['before'])} "
        f"after={_loads(load['after'])}"
    )
    empty = report["benchmark"]["empty"]["timing"]["clocks"]
    floor = " ".join(
        f"{short}_median_us={_us(empty[clock]['timing']['median_ns'])} "
        f"{short}_iqr_us={_us(empty[clock]['timing']['iqr_ns'])}"
        for clock, short in SHORT_CLOCKS.items()
    )
    print(f"{label} empty_kernel {floor}")
    for point in analysis["points"]:
        for side, short in SHORT_NAMES.items():
            entry = point["sides"][side]
            print(
                f"{label} point m={point['m']} n={point['n']} k={point['k']} "
                f"side={short} {_clocks(entry['clocks'])} "
                f"event_regime={entry['clocks']['mps_event']['regime']} "
                f"host_minus_event_us={_us(entry['host_minus_event']['value_ns'])} "
                "cross_domain=true"
            )
    for side, short in SHORT_NAMES.items():
        fits = " ".join(
            f"{name}_fixed_us={_us(analysis['fits'][side][clock]['intercept_ns'])} "
            f"{name}_residual={analysis['fits'][side][clock]['max_relative_residual']:.3f}"
            for clock, name in SHORT_CLOCKS.items()
        )
        print(f"{label} fit side={short} {fits}")


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    counts = (
        f"sweep_points={summary['sweep_points']} "
        f"timed_samples={summary['timed_samples']} scope={summary['scope']} "
        f"load_before={_loads(summary['load_average_before'])} "
        f"load_after={_loads(summary['load_average_after'])}"
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
    load_average: Callable[[], tuple[float, float, float]] | None = None,
) -> int:
    """Run one command and return its exit code; the keywords are for tests.

    A replaced backend or clock makes the report synthetic, never a
    measurement.
    """

    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            config = load_config(args.config) if args.config is not None else None
            _, summary = load_report(args.report, config=config)
            print(_summary_line("verified_gpu_timing", summary))
            return EXIT_OK
        _refuse_unusable_output(args.output, args.config)
        config = load_config(args.config)
        report = measure_report(
            config,
            backend=backend,
            clock=clock or time.perf_counter_ns,
            load_average=load_average or os.getloadavg,
        )
        summary = save_report(
            args.output, report, config=config, protected=(args.config,)
        )
    except GpuTimingUnavailableError as error:
        print(f"gpu-timing {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except GpuTimingVerificationError as error:
        print(f"gpu-timing {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"gpu-timing {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    _print_run(report)
    print(_summary_line("gpu_timing", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
