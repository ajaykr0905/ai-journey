"""Run and verify the Metal kernel lab on this host's Apple GPU.

``run`` loads a config, compiles the vector-add and grayscale kernels
through ``torch.mps.compile_shader``, checks them against PyTorch on MPS and
on the CPU, times both on the MPS device, verifies the sealed report, and
publishes it atomically without ever writing over the config. A backend or
clock injected through ``main`` yields a report printed and verified as
synthetic timing. ``verify`` recomputes a report on the CPU, optionally
against the config it must come from, and never imports torch.

Exit codes: 0 success; 1 a report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, refused outputs, and runs that failed
a check or exceeded their budget; 3 this host cannot run Metal kernels
through PyTorch (not macOS, no torch, no MPS device or no
``torch.mps.compile_shader``), so ``run`` stopped before it measured
anything.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .kernel_metal_config import load_config
from .kernel_metal_geometry import KernelMetalError
from .kernel_metal_io import load_report, save_report
from .kernel_metal_report import measure_report
from .kernel_metal_runtime import KernelMetalUnavailableError
from .kernel_metal_verify import KernelMetalVerificationError
from .memory_access_report_io import paths_alias

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "python -m ai_journey.kernel_metal_cli"


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Check and time Metal vector-add and grayscale kernels against "
            "PyTorch on this host's Apple GPU; no CUDA kernel or NVIDIA GPU is "
            "used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="check and time the kernels on this Apple GPU and seal a report",
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


def _refuse_unusable_output(output: Path, config: Path) -> None:
    """Refuse an output path the atomic writer would refuse, before running.

    A run compiles kernels and times the GPU for seconds, so an output that
    aliases the config or is a symlink, directory or other non-regular file
    fails first, and so does one whose nearest existing ancestor is not a
    writable directory.
    """

    if paths_alias(output, config):
        raise KernelMetalError("report output must not alias the config")
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise KernelMetalError("report output must be a regular non-symlink file")
    ancestor = output.parent
    # A dangling or looping link stops the walk too, then fails is_dir.
    while not (ancestor.exists() or ancestor.is_symlink()):
        if ancestor.parent == ancestor:
            break
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK | os.X_OK):
        raise KernelMetalError("report output must be inside a writable directory")


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _point_line(label: str, point: dict[str, object]) -> str:
    if point["kernel"] == "vector_add":
        extent = f"length={point['length']}"
    else:
        extent = f"height={point['height']} width={point['width']}"
    sides = " ".join(
        f"{side}_median_us={point[side]['median_ns_per_call'] / 1000:.2f} "
        f"{side}_iqr_us={point[side]['iqr_ns_per_call'] / 1000:.2f} "
        f"{side}_gb_per_second={point[side]['median_gb_per_second']:.2f}"
        for side in ("metal_kernel", "torch")
    )
    return (
        f"{label} benchmark kernel={point['kernel']} {extent} {sides} "
        f"torch_to_metal_time_ratio={point['torch_to_metal_time_ratio']:.2f}"
    )


def _print_run(report: dict[str, object]) -> None:
    label = _label(report)
    correctness = report["correctness"]
    summary = report["summary"]
    print(
        f"{label} correctness kernel=vector_add "
        f"cases={len(correctness['vector_add'])} "
        f"exact_cases={summary['vector_add_exact_cases']} "
        "implementations=metal_kernel/mps,torch/mps,torch/cpu"
    )
    print(
        f"{label} correctness kernel=rgb_to_gray "
        f"cases={len(correctness['rgb_to_gray'])} "
        "max_abs_difference_levels="
        f"{summary['rgb_to_gray_max_abs_difference_levels']} "
        f"metal_mismatched_pixels={summary['rgb_to_gray_metal_mismatched_pixels']} "
        f"bound_levels={correctness['bound_levels']}"
    )
    fused = report["shaders"]["contraction_probe"]["fused"]
    print(f"{label} contraction_probe fused={str(fused).lower()}")
    benchmark = report["benchmark"]
    for point in (*benchmark["vector_add"], *benchmark["rgb_to_gray"]):
        print(_point_line(label, point))


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    counts = (
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
            print(_summary_line("verified_kernel_metal", summary))
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
        print(f"kernel-metal {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except KernelMetalVerificationError as error:
        print(f"kernel-metal {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError) as error:
        print(f"kernel-metal {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    _print_run(report)
    print(_summary_line("kernel_metal", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
