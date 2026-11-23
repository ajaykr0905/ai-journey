"""Run and verify the Day 80 fused LLM-training kernels on this host's Apple GPU.

``run`` loads a config and a gpu-ceilings report, which the Day 74 verifier
must accept, prints the declared plan, compiles the library through
``torch.mps.compile_shader``, measures the math probes, checks every kernel
on the correctness cases, times each kernel with its PyTorch sequence on
the MPS device in one rotated run per point, verifies the sealed report and
publishes it atomically without ever writing over its inputs. A backend or
clock injected through ``main`` yields a report printed and verified as
synthetic timing. ``verify`` recomputes a report on the CPU, optionally
against the config and the gpu-ceilings report that it must come from, and
never imports torch.

Exit codes: 0 success; 1 a report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, refused outputs, and runs that failed
a check, hit a device error or exceeded their budget; 3 this host cannot run
Metal kernels through PyTorch (not macOS, no torch, no MPS device or no
``torch.mps.compile_shader``), so ``run`` stopped before it measured or
wrote anything. These are the codes of the Day 73 to 79 Metal labs.
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
from .metal_liger_config import load_config
from .metal_liger_io import load_report, read_ceiling, save_report
from .metal_liger_report import measure_report
from .metal_liger_runtime import MetalLigerUnavailableError
from .metal_liger_source import MetalLigerError
from .metal_liger_verify import MetalLigerVerificationError

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-metal-liger"
CLAIMS = (
    "cuda=false nvidia=false triton_gpu=false liger_code=false backward=false "
    "hardware_counters=false"
)


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Check and time fused float32 RMSNorm, SwiGLU and cross-entropy "
            "forward kernels against their PyTorch sequences on this host's "
            "Apple GPU, with a measured bandwidth ceiling; no CUDA kernel, "
            "NVIDIA GPU or Triton GPU kernel is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="check and time every Day 80 kernel on this Apple GPU",
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


def _shape(shape: dict[str, object]) -> str:
    return f"rows={shape['rows']} cols={shape['cols']}"


def plan_lines(label: str, plan: dict[str, object]) -> list[str]:
    """Print the plan that a report sealed: one line for each family."""

    return [
        f"{label} plan family={step['family']} kernel={step['kernel']} "
        f"base={step['base']} predicted={'|'.join(step['predicted'])} "
        f"kernel_sha256={step['source_sha256'][:12]}"
        for step in plan["steps"]
    ]


def _run_lines(report: dict[str, object]) -> list[str]:
    label = _label(report)
    load = report["host_load"]
    ceiling = report["ceiling"]
    lines = [
        f"{label} load_average before={_loads(load['before'])} "
        f"after={_loads(load['after'])}",
        f"{label} ceiling gb_per_second={ceiling['gb_per_second']:.2f} "
        f"kernel={ceiling['kernel']} buffer_bytes={ceiling['buffer_bytes']} "
        f"report={ceiling['report_sha256'][:12]} "
        f"measured={_flag(ceiling['measured'])} "
        f"load_before={_loads(ceiling['load_average_before'])}",
    ]
    for name, probe in report["math_probes"].items():
        lines.append(
            f"{label} probe function={name} max_ulps={probe['max_ulps']:.3f} "
            f"max_relative_error={probe['max_relative_error']:.3e} "
            f"holds={_flag(probe['holds'])}"
        )
    for case in report["correctness"]["cases"]:
        layout, output = case["layout"], case["output"]
        lines.append(
            f"{label} correctness family={case['family']} case={case['index']} "
            f"rows={layout['rows']} shortest={layout['shortest']} "
            f"longest={layout['longest']} group={layout['group_size']} "
            f"inputs={case['case']['inputs']} "
            f"bound_ratio={output['max_bound_ratio']:.3f} "
            f"identical_repeats={_flag(output['identical_repeats'])} "
            f"naive_finite={_flag(case['naive_float32_finite'])}"
        )
    for point in report["benchmark"]["points"]:
        sides = " ".join(
            f"{name}_gbps={point['sides'][name]['effective_gb_per_second']:.1f} "
            f"{name}_iqr="
            + "-".join(
                f"{rate:.1f}"
                for rate in point["sides"][name]["iqr_effective_gb_per_second"]
            )
            for name in point["side_order"]
        )
        calls = point["sides"][point["side_order"][0]]["calls_per_sample"]
        lines.append(
            f"{label} benchmark family={point['family']} {_shape(point['shape'])} "
            f"group={point['layout']['group_size']} calls_per_sample={calls} {sides}"
        )
    for row in report["speedup"]["points"]:
        lines.append(
            f"{label} speedup family={row['family']} {_shape(row['shape'])} "
            f"kernel_gbps={row['kernel']['effective_gb_per_second']:.1f} "
            f"torch_gbps={row['torch']['effective_gb_per_second']:.1f} "
            f"kernel_ceiling_fraction="
            f"{row['kernel']['effective_fraction_of_ceiling']:.3f} "
            f"ratio={row['to_torch']['ratio']:.3f} "
            f"verdict={row['to_torch']['verdict']} held={_flag(row['held'])}"
        )
    captured = report["warnings"]
    lines.append(
        f"{label} warnings count={captured['count']} distinct={captured['distinct']} "
        "torch_imported_before_capture="
        f"{_flag(captured['torch_imported_before_capture'])}"
    )
    return lines


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    held = sum(record["held"] for record in summary["hypotheses"].values())
    judged = sum(
        record["held"] + record["missed"] for record in summary["hypotheses"].values()
    )
    counts = (
        f"correctness_cases={summary['correctness_cases']} "
        f"benchmark_points={summary['benchmark_points']} "
        f"max_bound_ratio={max(summary['max_bound_ratio'].values()):.3f} "
        f"hypotheses_held={held}/{judged} "
        f"plan={summary['plan_sha256'][:12]} "
        f"ceiling={summary['ceiling_gb_per_second']:.2f} "
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
        raise MetalLigerError("report output must not alias the ceiling report")


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
            print(_summary_line("verified_metal_liger", summary))
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
    except MetalLigerUnavailableError as error:
        print(f"metal-liger {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except MetalLigerVerificationError as error:
        print(f"metal-liger {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"metal-liger {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in plan_lines(_label(report), report["plan"]):
        print(line)
    for line in _run_lines(report):
        print(line)
    print(_summary_line("metal_liger", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
