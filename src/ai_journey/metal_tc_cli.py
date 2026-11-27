"""Run and verify the Day 82 matmul kernels on this host's Apple GPU.

``run`` loads a config and a gpu-ceilings report, which the Day 74 verifier
must accept, prints the declared plan, compiles both libraries through
``torch.mps.compile_shader``, records the Metal language version and the
bfloat support, checks the four kernels on the correctness cases, times S1,
M1, M2, M3 and ``torch.matmul`` in three dtypes in one rotated run per
size, verifies the sealed report and publishes it atomically without ever
writing over its inputs. A backend or clock injected through ``main``
yields a report printed and verified as synthetic timing. ``verify``
recomputes a report on the CPU, optionally against the config and the
gpu-ceilings report that it must come from, and never imports torch.

Exit codes: 0 success; 1 a report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, refused outputs, and runs that failed
a check, hit a device error or exceeded their budget; 3 this host cannot run
Metal kernels through PyTorch (not macOS, no torch, no MPS device or no
``torch.mps.compile_shader``), so ``run`` stopped before it measured or
wrote anything. These are the codes of the Day 73 to 81 Metal labs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .kernel_metal_cli import _refuse_unusable_output
from .memory_access_report_io import paths_alias
from .metal_tc_config import load_config
from .metal_tc_io import load_report, read_ceiling, save_report
from .metal_tc_report import measure_report
from .metal_tc_runtime import MetalTcUnavailableError
from .metal_tc_source import MetalTcError
from .metal_tc_verify import MetalTcVerificationError

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-metal-tc"
CLAIMS = "tensor_cores=false cuda=false nvidia=false ptx=false triton_gpu=false"


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Check and time a tiled matmul with and without the simdgroup matrix "
            "path in float32, half and bfloat against torch.matmul on this host's "
            "Apple GPU, with measured ceilings; Apple GPUs have no NVIDIA Tensor "
            "Cores, and no CUDA kernel or NVIDIA GPU is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="check and time every Day 82 side on this Apple GPU",
        allow_abbrev=False,
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--ceilings", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify",
        help="recompute a sealed report on the CPU, without torch",
        allow_abbrev=False,
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    verify.add_argument("--ceilings", type=Path)
    return parser


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def _flag(value: bool | None) -> str:
    return "none" if value is None else "true" if value else "false"


def plan_lines(label: str, plan: dict[str, object]) -> list[str]:
    """Print the plan that a report sealed: one line for each step."""

    return [
        f"{label} plan step={step['step']} first={step['first']} "
        f"base={step['base']} predicted={'|'.join(step['predicted'])}"
        for step in plan["steps"]
    ]


def _run_lines(report: dict[str, object]) -> list[str]:
    label = _label(report)
    load = report["host_load"]
    ceiling = report["ceiling"]
    gains = report["speedup"]
    support = report["support"]
    lines = [
        f"{label} load_average before={_loads(load['before'])} "
        f"after={_loads(load['after'])}",
        f"{label} ceiling gb_per_second={ceiling['gb_per_second']:.2f} "
        f"fma_gflops_per_second={gains['fma_gflops_per_second']:.1f} "
        f"ridge_flops_per_byte={gains['ridge_flops_per_byte']:.1f} "
        f"report={ceiling['report_sha256'][:12]} "
        f"measured={_flag(ceiling['measured'])} "
        f"load_before={_loads(ceiling['load_average_before'])}",
        f"{label} support metal_language_version="
        f"{support['metal_language_version']} "
        + " ".join(
            f"{name}={_flag(record['supported'])}:{record['reason'] or 'none'}"
            for name, record in support["sides"].items()
        ),
    ]
    for case in report["correctness"]["cases"]:
        shape = case["case"]
        lines.append(
            f"{label} correctness case={case['index']} m={shape['m']} "
            f"n={shape['n']} k={shape['k']} "
            + " ".join(
                f"{name}_bound_ratio={entry['max_bound_ratio']:.2e} "
                f"{name}_relative={entry['max_relative_error']:.2e} "
                f"{name}_identical={_flag(entry['identical_repeats'])}"
                for name, entry in case["outputs"].items()
            )
            + f" m1_same_bits_as_s1={_flag(case['m1_same_bits_as_s1'])}"
        )
    for point, row in zip(report["benchmark"]["points"], gains["points"], strict=True):
        sides = " ".join(
            f"{name}_tflops={entry['tflops_per_second']:.3f} "
            f"{name}_iqr_ms={entry['iqr_ns_per_call'] / 1e6:.3f} "
            f"{name}_of_fma={row['rates'][name]['fraction_of_fma']:.3f}"
            for name, entry in point["sides"].items()
        )
        calls = point["sides"]["s1"]["calls_per_sample"]
        lines.append(
            f"{label} benchmark point={point['index']} size={point['size']} "
            f"calls_per_sample={calls} {sides} "
            f"m1_same_bits_as_s1={_flag(point['m1_same_bits_as_s1'])} "
            "torch_float32_same_bits_as_s1="
            f"{_flag(point['torch_float32_same_bits_as_s1'])}"
        )
        lines.append(
            f"{label} verdicts point={row['index']} size={row['size']} "
            + " ".join(
                f"{step['step']}={step['ratio']:.3f}:{step['verdict']}:"
                f"{'held' if step['held'] else 'missed'}"
                if step["ran"]
                else f"{step['step']}=not_run"
                for step in row["steps"]
            )
        )
    captured = report["warnings"]
    lines.append(
        f"{label} warnings count={captured['count']} distinct={captured['distinct']} "
        "torch_imported_before_capture="
        f"{_flag(captured['torch_imported_before_capture'])}"
    )
    return lines


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    records = summary["hypotheses"].values()
    held = sum(record["held"] for record in records)
    judged = sum(record["held"] + record["missed"] for record in records)
    counts = (
        f"correctness_cases={summary['correctness_cases']} "
        f"benchmark_points={summary['benchmark_points']} "
        f"max_bound_ratio={max(summary['max_bound_ratio'].values()):.3e} "
        f"hypotheses_held={held}/{judged} "
        f"plan={summary['plan_sha256'][:12]} "
        f"metal_language_version={summary['metal_language_version']} "
        f"bfloat_kernel={_flag(summary['supported']['m3'])} "
        f"fma={summary['fma_gflops_per_second']:.1f} "
        f"bandwidth={summary['bandwidth_gb_per_second']:.2f} "
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


def _refuse_inputs(output: Path, config: Path, ceilings: Path) -> None:
    _refuse_unusable_output(output, config)
    if paths_alias(output, ceilings):
        raise MetalTcError("report output must not alias the ceiling report")


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
            ceiling = read_ceiling(args.ceilings) if args.ceilings is not None else None
            _, summary = load_report(args.report, config=config, ceiling=ceiling)
            print(_summary_line("verified_metal_tc", summary))
            return EXIT_OK
        _refuse_inputs(args.output, args.config, args.ceilings)
        config = load_config(args.config)
        ceiling = read_ceiling(args.ceilings)
        report = measure_report(
            config,
            ceiling,
            backend=backend,
            clock=clock or time.perf_counter_ns,
            load_average=load_average or os.getloadavg,
        )
        summary = save_report(
            args.output,
            report,
            config=config,
            ceiling=ceiling,
            protected=(args.config, args.ceilings),
        )
    except MetalTcUnavailableError as error:
        print(f"metal-tc {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except MetalTcVerificationError as error:
        print(f"metal-tc {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"metal-tc {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in plan_lines(_label(report), report["plan"]):
        print(line)
    for line in _run_lines(report):
        print(line)
    print(_summary_line("metal_tc", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
