"""Run and verify the Day 81 attention kernels on this host's Apple GPU.

``run`` loads a config and a gpu-ceilings report, which the Day 74 verifier
must accept, prints the declared plan, compiles the library through
``torch.mps.compile_shader``, measures the exp probe, checks N0 and F1 on
the correctness cases, measures the memory of each side, times N0, F1 and
the PyTorch call on the MPS device in one rotated run per point, verifies
the sealed report and publishes it atomically without ever writing over its
inputs. A backend or clock injected through ``main`` yields a report
printed and verified as synthetic timing. ``verify`` recomputes a report on
the CPU, optionally against the config and the gpu-ceilings report that it
must come from, and never imports torch.

Exit codes: 0 success; 1 a report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, refused outputs, and runs that failed
a check, hit a device error or exceeded their budget; 3 this host cannot run
Metal kernels through PyTorch (not macOS, no torch, no MPS device or no
``torch.mps.compile_shader``), so ``run`` stopped before it measured or
wrote anything. These are the codes of the Day 73 to 80 Metal labs.
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
from .metal_flash_config import load_config
from .metal_flash_io import load_report, read_ceiling, save_report
from .metal_flash_report import measure_report
from .metal_flash_runtime import MetalFlashUnavailableError
from .metal_flash_source import MetalFlashError
from .metal_flash_verify import MetalFlashVerificationError

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-metal-flash"
CLAIMS = "cuda=false nvidia=false ptx=false triton_gpu=false backward=false"
_MIB = 1 << 20


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Check, measure and time naive and flash float32 attention forward "
            "kernels against torch.nn.functional.scaled_dot_product_attention "
            "on this host's Apple GPU, with measured ceilings; no CUDA kernel, "
            "NVIDIA GPU or Triton GPU kernel is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="check, measure and time every Day 81 side on this Apple GPU",
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
    probe = report["exp_probe"]
    lines = [
        f"{label} load_average before={_loads(load['before'])} "
        f"after={_loads(load['after'])}",
        f"{label} ceiling gb_per_second={ceiling['gb_per_second']:.2f} "
        f"fma_gflops_per_second="
        f"{report['compute_ceiling']['fma_gflops_per_second']:.1f} "
        f"report={ceiling['report_sha256'][:12]} "
        f"measured={_flag(ceiling['measured'])} "
        f"load_before={_loads(ceiling['load_average_before'])}",
        f"{label} probe function=exp max_ulps={probe['max_ulps']:.3f} "
        f"holds={_flag(probe['holds'])}",
    ]
    for case in report["correctness"]["cases"]:
        layout = case["layout"]
        for mode, record in case["modes"].items():
            outputs = record["outputs"]
            lines.append(
                f"{label} correctness case={case['index']} mode={mode} "
                f"sequences={layout['sequences']} shortest={layout['shortest']} "
                f"longest={layout['longest']} inputs={case['case']['inputs']} "
                + " ".join(
                    f"{name}_bound_ratio={entry['max_bound_ratio']:.2e} "
                    f"{name}_identical={_flag(entry['identical_repeats'])}"
                    for name, entry in outputs.items()
                )
                + f" unscaled_exp_overflows={_flag(record['unscaled_exp_overflows'])}"
            )
    for point in report["benchmark"]["points"]:
        sides = " ".join(
            f"{name}_ms={entry['median_ns_per_call'] / 1e6:.3f} "
            f"{name}_iqr_ms={entry['iqr_ns_per_call'] / 1e6:.3f} "
            f"{name}_gflops={entry['gflops_per_second']:.1f}"
            for name, entry in point["sides"].items()
        )
        memory = " ".join(
            f"{name}_peak_driver_mib="
            f"{entry['peak_driver_over_before_bytes'] / _MIB:.1f} "
            f"{name}_construction_mib={entry['known_construction_bytes'] / _MIB:.1f}"
            for name, entry in point["memory"].items()
        )
        calls = point["sides"]["n0"]["calls_per_sample"]
        lines.append(
            f"{label} benchmark point={point['index']} "
            f"sequences={point['sequences']} length={point['length']} "
            f"mode={point['mode']} calls_per_sample={calls} {sides} {memory}"
        )
    for row in report["speedup"]["points"]:
        lines.append(
            f"{label} verdicts point={row['index']} length={row['length']} "
            f"mode={row['mode']} "
            + " ".join(
                f"{step['step']}={step['ratio']:.3f}:{step['verdict']}:"
                f"{'held' if step['held'] else 'missed'}"
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
    held = sum(record["held"] for record in summary["hypotheses"].values())
    judged = sum(
        record["held"] + record["missed"] for record in summary["hypotheses"].values()
    )
    counts = (
        f"correctness_cases={summary['correctness_cases']} "
        f"benchmark_points={summary['benchmark_points']} "
        f"max_bound_ratio={max(summary['max_bound_ratio'].values()):.3e} "
        f"hypotheses_held={held}/{judged} "
        f"plan={summary['plan_sha256'][:12]} "
        f"bandwidth={summary['bandwidth_gb_per_second']:.2f} "
        f"fma={summary['fma_gflops_per_second']:.1f} "
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
        raise MetalFlashError("report output must not alias the ceiling report")


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
            print(_summary_line("verified_metal_flash", summary))
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
    except MetalFlashUnavailableError as error:
        print(f"metal-flash {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except MetalFlashVerificationError as error:
        print(f"metal-flash {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"metal-flash {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in plan_lines(_label(report), report["plan"]):
        print(line)
    for line in _run_lines(report):
        print(line)
    print(_summary_line("metal_flash", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
