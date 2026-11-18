"""Capture, analyze and verify torch.profiler traces of the Day 74 matmul kernels.

``capture`` loads a config, profiles both naive Metal kernels and
``torch.matmul`` on this host's Apple GPU through the MPS backend, seals a
report, verifies it, and publishes it with its trace files in one output
directory. ``analyze`` reads any trace file that ``torch.profiler`` exported
and prints its top ops by self time, its top-level gaps, the host time per
iteration and a text timeline. ``verify`` checks a report against the traces
beside it. ``analyze`` and ``verify`` never import torch.

``ProfilerActivity.CPU`` records only host-side dispatch and the host's wait
in ``torch.mps.synchronize()``; it records no GPU kernel time. The named
bottleneck is an inference from those timings and the Day 74 ceilings, not
a hardware counter reading.

Exit codes: 0 success; 1 a report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, refused outputs, and captures that
failed a check or hit a device error; 3 this host cannot run Metal kernels
through PyTorch (not macOS, no torch, no MPS device or no
``torch.mps.compile_shader``), so ``capture`` stopped before it profiled
anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .matmul_metal_runtime import MatmulMetalUnavailableError
from .memory_access_report_io import paths_alias
from .profile_trace_analysis import (
    MAX_TOP_N,
    iteration_times,
    median_low,
    op_stats,
    top_level_gaps,
    top_ops,
)
from .profile_trace_config import load_config
from .profile_trace_events import ProfileTraceError, load_trace
from .profile_trace_io import REPORT_FILE, load_report, save_capture
from .profile_trace_layout import ITERATION
from .profile_trace_report import measure_report
from .profile_trace_timeline import MAX_ROWS, render_timeline
from .profile_trace_verify import ProfileTraceVerificationError

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-profile-trace"
SHORT_NAMES = {
    "matmul_x_columns": "x_columns",
    "matmul_x_rows": "x_rows",
    "torch_matmul": "torch",
}


def build_parser() -> argparse.ArgumentParser:
    """Define the ``capture``, ``analyze`` and ``verify`` commands."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Profile the Day 74 naive Metal matmul kernels and torch.matmul with "
            "torch.profiler on this host's Apple GPU, and analyze the traces; no "
            "CUDA kernel, NVIDIA GPU or Nsight tool is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser(
        "capture",
        help="profile the kernels on this Apple GPU and seal a report",
        allow_abbrev=False,
    )
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    analyze = commands.add_parser(
        "analyze",
        help="analyze one torch.profiler Chrome trace, without torch",
        allow_abbrev=False,
    )
    analyze.add_argument("--trace", type=Path, required=True)
    analyze.add_argument("--top", type=int, default=8)
    analyze.add_argument("--width", type=int, default=96)
    analyze.add_argument("--depth", type=int, default=3, choices=range(1, MAX_ROWS + 1))
    analyze.add_argument("--iteration-name", default=ITERATION)
    verify = commands.add_parser(
        "verify",
        help="check a report against the trace files beside it, without torch",
        allow_abbrev=False,
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    return parser


def _refuse_output(directory: Path, config: Path) -> None:
    """Refuse an output the publisher would refuse, before the GPU runs."""

    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ProfileTraceError("capture output must be a real directory")
    if paths_alias(directory / REPORT_FILE, config):
        raise ProfileTraceError("capture output must not alias the config")


def _us(nanoseconds: float) -> str:
    return f"{nanoseconds / 1e3:.1f}"


def _analyze(args: argparse.Namespace) -> None:
    trace = load_trace(args.trace)
    stats = op_stats(trace)
    print(f"trace events={trace.event_count} spans={len(trace.spans)} ops={len(stats)}")
    for rank, row in enumerate(top_ops(stats, min(args.top, MAX_TOP_N)), start=1):
        print(
            f"top {rank} name={row.name} calls={row.calls} self_us={_us(row.self_ns)} "
            f"total_us={_us(row.total_ns)}"
        )
    for thread in top_level_gaps(trace):
        print(
            f"gaps pid={thread.pid} tid={thread.tid} calls={thread.calls} "
            f"busy_us={_us(thread.busy_ns)} gap_us={_us(thread.total_gap_ns)} "
            f"max_gap_us={_us(max((g.gap_ns for g in thread.gaps), default=0))}"
        )
    iterations = iteration_times(trace, args.iteration_name)
    if iterations:
        hosts = tuple(item.host_ns for item in iterations)
        print(
            f"iterations name={args.iteration_name} count={len(hosts)} "
            f"median_low_us={_us(median_low(hosts))} min_us={_us(min(hosts))} "
            f"max_us={_us(max(hosts))}"
        )
    print(render_timeline(trace, width=args.width, depth_limit=args.depth), end="")


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _print_capture(report: dict[str, object]) -> None:
    label = _label(report)
    loads = report["load_average"]
    print(
        f"{label} load_before={','.join(map(str, loads['before']))} "
        f"load_after={','.join(map(str, loads['after']))}"
    )
    for entry in report["shapes"]:
        found = entry["bottleneck"]
        print(
            f"{label} shape m={entry['m']} n={entry['n']} k={entry['k']} "
            f"bottleneck={found['top_op']} kind={found['kind']} "
            f"share={found['share_of_iteration_host_time']:.3f}"
        )
        for side in found["sides"]:
            print(
                f"{label} side m={entry['m']} {SHORT_NAMES[side['side']]} "
                f"host_us={_us(side['median_host_ns'])} "
                f"dispatch_us={_us(side['median_dispatch_ns'])} "
                f"synchronize_us={_us(side['median_synchronize_ns'])} "
                f"device_event_us={_us(side['median_device_event_ns'])} "
                f"host_gflops={side['host_observed_gflops_per_second']:.1f} "
                f"fma_fraction={side['fraction_of_fma_ceiling']:.4f}"
            )


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    kind = "measurement" if summary["measured"] else "synthetic_timing"
    return (
        f"{prefix}_{kind}={summary['report_sha256']} shapes={summary['shapes']} "
        f"scope={summary['scope']} bottlenecks={','.join(summary['bottlenecks'])} "
        "gpu_kernels_recorded=false cuda=false nvidia=false"
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    session: object | None = None,
    load_reader: Callable[[], tuple[float, ...]] = os.getloadavg,
) -> int:
    """Run one command and return its exit code; ``session`` is for tests.

    A replaced session makes the report synthetic, never a measurement.
    """

    args = build_parser().parse_args(argv)
    try:
        if args.command == "analyze":
            _analyze(args)
            return EXIT_OK
        if args.command == "verify":
            config = load_config(args.config) if args.config is not None else None
            _, summary = load_report(args.report, config=config)
            print(_summary_line("verified_profile_trace", summary))
            return EXIT_OK
        _refuse_output(args.output_dir, args.config)
        config = load_config(args.config)
        report, captured = measure_report(
            config, session=session, load_reader=load_reader
        )
        summary = save_capture(
            args.output_dir, report, captured, config=config, protected=(args.config,)
        )
    except MatmulMetalUnavailableError as error:
        print(f"profile-trace {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except ProfileTraceVerificationError as error:
        print(f"profile-trace {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"profile-trace {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    _print_capture(report)
    print(f"report {args.output_dir / REPORT_FILE}")
    print(_summary_line("profile_trace", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
