"""Run and verify the naive Metal matmul lab on this host's Apple GPU.

``run`` loads a config, compiles both naive kernels through
``torch.mps.compile_shader``, checks them against a float64 reference and
against ``torch.matmul`` on MPS and on the CPU, times all three on the MPS
device, verifies the sealed report, and publishes it atomically without ever
writing over the config. A backend or clock injected through ``main`` yields
a report printed and verified as synthetic timing. ``verify`` recomputes a
report on the CPU, optionally against the config it must come from, and
never imports torch.

Exit codes: 0 success; 1 a report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, refused outputs, and runs that failed
a check, hit a device error or exceeded their budget; 3 this host cannot run
Metal kernels through PyTorch (not macOS, no torch, no MPS device or no
``torch.mps.compile_shader``), so ``run`` stopped before it measured
anything. PyTorch raises ``RuntimeError`` when the MPS device fails, for
example when it runs out of memory; that is exit 2, so exit 1 keeps its one
meaning.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .kernel_metal_cli import _refuse_unusable_output
from .matmul_metal_config import load_config
from .matmul_metal_io import load_report, save_report
from .matmul_metal_report import measure_report
from .matmul_metal_runtime import MatmulMetalUnavailableError
from .matmul_metal_verify import MatmulMetalVerificationError

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-matmul-metal"
SHORT_NAMES = {
    "matmul_x_columns": "x_columns",
    "matmul_x_rows": "x_rows",
    "torch_matmul": "torch",
}


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Check and time two naive float32 Metal matmul kernels against "
            "torch.matmul on this host's Apple GPU; no CUDA kernel or NVIDIA GPU "
            "is used."
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


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _case_line(label: str, case: dict[str, object]) -> str:
    errors = {
        f"{SHORT_NAMES[output['implementation']]}_{output['device']}": output[
            "max_relative_error"
        ]
        for output in case["outputs"]
    }
    return (
        f"{label} correctness m={case['m']} n={case['n']} k={case['k']} "
        f"naive_error={errors['x_columns_mps']:.3g} "
        f"torch_mps_error={errors['torch_mps']:.3g} "
        f"torch_cpu_error={errors['torch_cpu']:.3g} bound={case['bound']:.3g} "
        "naive_same_bits=true"
    )


def _point_line(label: str, point: dict[str, object]) -> str:
    sides = " ".join(
        f"{short}_median_us={point['sides'][side]['median_ns_per_call'] / 1e3:.1f} "
        f"{short}_iqr_us={point['sides'][side]['iqr_ns_per_call'] / 1e3:.1f} "
        f"{short}_gflops={point['sides'][side]['median_gflops_per_second']:.1f}"
        for side, short in SHORT_NAMES.items()
    )
    ratios = point["naive_to_torch_gflops_ratio"]
    return (
        f"{label} benchmark m={point['m']} n={point['n']} k={point['k']} {sides} "
        f"x_columns_to_torch={ratios['matmul_x_columns']:.3f} "
        f"x_rows_to_torch={ratios['matmul_x_rows']:.3f}"
    )


def _ceiling_line(label: str, benchmark: dict[str, object]) -> str:
    ceiling = benchmark["ceiling"]
    top = benchmark["points"][ceiling["ceiling_index"]]
    naive = " ".join(
        f"{SHORT_NAMES[name]}_best_gflops={entry['best_median_gflops_per_second']:.1f} "
        f"{SHORT_NAMES[name]}_fraction={entry['fraction_of_measured_ceiling']:.3f}"
        for name, entry in ceiling["naive"].items()
    )
    return (
        f"{label} ceiling torch_gflops="
        f"{ceiling['measured_ceiling_gflops_per_second']:.1f} "
        f"at m={top['m']} n={top['n']} k={top['k']} {naive}"
    )


def _print_run(report: dict[str, object]) -> None:
    label = _label(report)
    for case in report["correctness"]["cases"]:
        print(_case_line(label, case))
    for point in report["benchmark"]["points"]:
        print(_point_line(label, point))
    print(_ceiling_line(label, report["benchmark"]))


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
            print(_summary_line("verified_matmul_metal", summary))
            return EXIT_OK
        _refuse_unusable_output(args.output, args.config)
        config = load_config(args.config)
        report = measure_report(
            config, backend=backend, clock=clock or time.perf_counter_ns
        )
        summary = save_report(
            args.output, report, config=config, protected=(args.config,)
        )
    except MatmulMetalUnavailableError as error:
        print(f"matmul-metal {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except MatmulMetalVerificationError as error:
        print(f"matmul-metal {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"matmul-metal {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    _print_run(report)
    print(_summary_line("matmul_metal", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
