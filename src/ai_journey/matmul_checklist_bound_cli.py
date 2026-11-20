"""Time one call of each checklist side on the three Day 75 clocks.

``timing`` loads a checklist config, compiles the Day 74 naive source and
the checklist library through ``torch.mps.compile_shader``, times one call
of every side on the three Day 75 clocks at a 1 x 1 x 1 floor and at every
sweep shape, verifies the sealed report, and publishes it atomically without
ever writing over the config. A backend or clock injected through ``main``
yields a report printed and verified as synthetic timing.
``verify-timing`` recomputes a timing report on the CPU, optionally against
the config it must come from, and never imports torch.

Exit codes: 0 success; 1 a report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, refused outputs, and runs that failed
a check, hit a device error or exceeded their budget; 3 this host cannot run
Metal kernels through PyTorch, so ``timing`` stopped before it measured or
wrote anything. These are the codes of the Day 73 to 77 Metal labs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .kernel_metal_cli import _refuse_unusable_output
from .matmul_checklist_bound_timing import measure_timing_report
from .matmul_checklist_bound_timing_io import load_timing_report, save_timing_report
from .matmul_checklist_bound_traffic import MatmulChecklistBoundVerificationError
from .matmul_checklist_cli import short_name
from .matmul_checklist_config import load_config
from .matmul_checklist_runtime import MatmulChecklistUnavailableError

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
CLAIMS = "cuda=false nvidia=false cublas=false hardware_counters=false"


def build_parser() -> argparse.ArgumentParser:
    """Define every command with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Time one call of each Day 77 checklist side on this host's Apple "
            "GPU; no CUDA kernel, NVIDIA GPU, cuBLAS or hardware counter is used."
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

    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify-timing":
            config = load_config(args.config) if args.config is not None else None
            _, summary = load_timing_report(args.report, config=config)
            lines = [timing_summary_line("verified_matmul_checklist_one_call", summary)]
        else:
            lines = _timing(args, backend, clock, load_average)
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
