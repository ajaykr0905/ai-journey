"""Run and verify the tiled Metal matmul lab on this host's Apple GPU.

``run`` loads a config, compiles the Day 74 naive source and the tiled
library through ``torch.mps.compile_shader``, checks the naive kernel and
every admitted tile on the correctness cases, times them with
``torch.matmul`` on the MPS device, verifies the sealed report, and
publishes it atomically without ever writing over the config. A backend or
clock injected through ``main`` yields a report printed and verified as
synthetic timing. ``verify`` recomputes a report on the CPU, optionally
against the config it must come from, and never imports torch.

Exit codes: 0 success; 1 a report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, refused outputs, and runs that failed
a check, hit a device error or exceeded their budget; 3 this host cannot run
Metal kernels through PyTorch (not macOS, no torch, no MPS device or no
``torch.mps.compile_shader``), so ``run`` stopped before it measured or
wrote anything. These are the codes of the Day 73 to 75 Metal labs, so
scripts treat them alike.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .kernel_metal_cli import _refuse_unusable_output
from .matmul_tiled_config import load_config
from .matmul_tiled_io import load_report, save_report
from .matmul_tiled_report import measure_report
from .matmul_tiled_runtime import MatmulTiledUnavailableError
from .matmul_tiled_verify import MatmulTiledVerificationError

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-matmul-tiled"
SHORT_NAMES = {"matmul_x_columns": "naive", "torch_matmul": "torch"}


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Check and time the naive Day 74 kernel, three tiled float32 Metal "
            "matmul kernels and torch.matmul on this host's Apple GPU; no CUDA "
            "kernel, NVIDIA GPU or cuBLAS is used."
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


def short_name(name: str) -> str:
    """Spell a side as the CLI prints it: ``naive``, ``tiled_16`` or ``torch``."""

    return SHORT_NAMES.get(name, name.removeprefix("matmul_"))


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _step(step: dict[str, object] | None) -> str:
    return "none" if step is None else f"{step['ratio']:.3f}:{step['verdict']}"


def _case_line(label: str, case: dict[str, object]) -> str:
    outputs = " ".join(
        f"{short_name(output['kernel'])}_error={output['max_relative_error']:.3g} "
        f"{short_name(output['kernel'])}_replay_bits={_flag(output['replay_bits'])}"
        for output in case["outputs"]
    )
    return (
        f"{label} correctness m={case['m']} n={case['n']} k={case['k']} "
        f"bound={case['bound']:.3g} {outputs}"
    )


def _point_line(label: str, point: dict[str, object]) -> str:
    sides = " ".join(
        f"{short_name(side)}_gflops="
        f"{point['sides'][side]['median_gflops_per_second']:.1f} "
        f"{short_name(side)}_iqr="
        + "-".join(
            f"{rate:.1f}" for rate in point["sides"][side]["iqr_gflops_per_second"]
        )
        for side in point["side_order"]
    )
    return f"{label} benchmark m={point['m']} n={point['n']} k={point['k']} {sides}"


def _numbers_line(label: str, entry: dict[str, object]) -> str:
    tiled = entry["tiled"]
    chosen = (
        "none"
        if tiled is None
        else f"{short_name(tiled['side'])}:{tiled['median_gflops_per_second']:.1f}"
    )
    return (
        f"{label} three_numbers m={entry['m']} n={entry['n']} k={entry['k']} "
        f"naive_gflops={entry['naive']['median_gflops_per_second']:.1f} "
        f"tiled={chosen} "
        f"vendor_gflops={entry['vendor']['median_gflops_per_second']:.1f} "
        "cublas=not_measured "
        f"tiled_over_naive={_step(entry['tiled_over_naive'])} "
        f"vendor_over_tiled={_step(entry['vendor_over_tiled'])}"
    )


def _print_run(report: dict[str, object]) -> None:
    label = _label(report)
    load = report["host_load"]
    print(
        f"{label} load_average before={_loads(load['before'])} "
        f"after={_loads(load['after'])}"
    )
    for refusal in report["refusals"]:
        print(
            f"{label} refused variant={refusal['name']} reason={refusal['reason']} "
            f"allowed_threads={refusal['allowed_threads']} "
            f"static_threadgroup_bytes={refusal['static_threadgroup_bytes']}"
        )
    for case in report["correctness"]["cases"]:
        print(_case_line(label, case))
    for point in report["benchmark"]["points"]:
        print(_point_line(label, point))
    for entry in report["comparison"]["points"]:
        print(_numbers_line(label, entry))
    captured = report["warnings"]
    print(
        f"{label} warnings count={captured['count']} distinct={captured['distinct']} "
        "torch_imported_before_capture="
        f"{_flag(captured['torch_imported_before_capture'])}"
    )


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    refused = ",".join(summary["refused_variants"]) or "none"
    counts = (
        f"correctness_cases={summary['correctness_cases']} "
        f"benchmark_points={summary['benchmark_points']} refused={refused} "
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
            print(_summary_line("verified_matmul_tiled", summary))
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
    except MatmulTiledUnavailableError as error:
        print(f"matmul-tiled {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except MatmulTiledVerificationError as error:
        print(f"matmul-tiled {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"matmul-tiled {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    _print_run(report)
    print(_summary_line("matmul_tiled", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
