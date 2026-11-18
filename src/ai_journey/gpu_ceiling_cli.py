"""Run and verify the measured-ceilings lab on this host's Apple GPU.

``run`` loads a config, compiles the FMA, stream and bank kernels through
``torch.mps.compile_shader``, checks each one against its exact CPU rebuild,
times the three sweeps on the MPS device, derives the ceilings, verifies the
sealed report, and publishes it atomically without ever writing over the
config. Every line it prints carries the measurement label and the run's
load averages are part of the report. A backend, clock or load reader
injected through ``main`` yields a report printed and verified as synthetic
timing. ``verify`` recomputes a report on the CPU, optionally against the
config it must come from, and never imports torch.

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
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .gpu_ceiling_config import load_config
from .gpu_ceiling_io import load_report, save_report
from .gpu_ceiling_report import measure_report
from .gpu_ceiling_runtime import GpuCeilingUnavailableError
from .gpu_ceiling_verify import GpuCeilingVerificationError
from .kernel_metal_cli import _refuse_unusable_output

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-gpu-ceilings"


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Measure compute, bandwidth and threadgroup-memory ceilings on this "
            "host's Apple GPU; no vendor peak, CUDA kernel or NVIDIA GPU is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="check and time the ceiling kernels on this Apple GPU, seal a report",
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


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def _run_lines(report: dict[str, object]) -> list[str]:
    label = _label(report)
    load = report["host_load"]
    correctness, benchmark = report["correctness"], report["benchmark"]
    ceilings = report["ceilings"]
    lines = [
        f"{label} host_load before={_loads(load['before'])} "
        f"after={_loads(load['after'])}",
        f"{label} correctness fma_cases={len(correctness['fma'])} "
        f"stream_cases={len(correctness['stream'])} "
        f"bank_cases={len(correctness['bank'])} exact=true",
    ]
    for point in benchmark["fma"]:
        lines.append(
            f"{label} fma chains={point['chains']} fma_steps={point['fma_steps']} "
            f"median_us={point['median_ns_per_call'] / 1000:.2f} "
            f"iqr_us={point['iqr_ns_per_call'] / 1000:.2f} "
            f"gflops_per_second={point['median_gflops_per_second']:.1f}"
        )
    for point in benchmark["stream"]:
        lines.append(
            f"{label} stream kernel={point['kernel']} "
            f"buffer_bytes={point['buffer_bytes']} "
            f"calls_per_sample={point['calls_per_sample']} "
            f"median_us={point['median_ns_per_call'] / 1000:.2f} "
            f"gb_per_second={point['median_gb_per_second']:.2f}"
        )
    rows = {row["stride"]: row for row in ceilings["bank"]["rows"]}
    for point in benchmark["bank"]:
        row = rows[point["stride"]]
        ratio = (
            row["unpadded_relative_to_stride_1"]
            if point["row_words"] == 32
            else row["padded_relative_to_unpadded_stride_1"]
        )
        lines.append(
            f"{label} bank row_words={point['row_words']} stride={point['stride']} "
            f"ps_per_access={point['median_ps_per_access']:.3f} "
            f"relative_to_unpadded_stride_1={ratio:.2f}"
        )
    compute, bandwidth, bank = ceilings["fma"], ceilings["bandwidth"], ceilings["bank"]
    lines += [
        f"{label} ceiling compute gflops_per_second={compute['gflops_per_second']:.1f} "
        f"chains={compute['chains']} fma_steps={compute['fma_steps']} "
        f"fixed_share={compute['fixed_share_of_ceiling_call']:.4f} "
        f"scales_with_work={str(compute['scales_with_work']).lower()}",
        f"{label} ceiling bandwidth gb_per_second={bandwidth['gb_per_second']:.2f} "
        f"kernel={bandwidth['kernel']} buffer_bytes={bandwidth['buffer_bytes']}",
        f"{label} ceiling ridge_flops_per_byte="
        f"{ceilings['ridge_point_flops_per_byte']:.2f}",
        f"{label} bank finding={bank['finding']} "
        f"largest_unpadded_ratio={bank['largest_unpadded_ratio']:.2f} "
        f"largest_padded_ratio={bank['largest_padded_ratio']:.2f}",
    ]
    return lines


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    counts = (
        f"correctness_cases={summary['correctness_cases']} "
        f"fma_points={summary['fma_points']} "
        f"stream_points={summary['stream_points']} "
        f"bank_points={summary['bank_points']} scope={summary['scope']}"
    )
    claims = "vendor_peak=false cuda=false nvidia=false"
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
            print(_summary_line("verified_gpu_ceiling", summary))
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
    except GpuCeilingUnavailableError as error:
        print(f"gpu-ceiling {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except GpuCeilingVerificationError as error:
        print(f"gpu-ceiling {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"gpu-ceiling {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in _run_lines(report):
        print(line)
    print(_summary_line("gpu_ceiling", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
