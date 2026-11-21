"""Run and verify the Day 79 fused softmax lab on this host's Apple GPU.

``run`` loads a config and a gpu-ceilings report, which the Day 74 verifier
must accept, prints the declared plan, compiles the softmax library through
``torch.mps.compile_shader``, measures the exp probe, checks every step on
the correctness cases, times every step with ``torch.softmax`` on the MPS
device in one rotated run, verifies the sealed report and publishes it
atomically without ever writing over its inputs. A backend or clock
injected through ``main`` yields a report printed and verified as synthetic
timing. ``verify`` recomputes a report on the CPU, optionally against the
config and the gpu-ceilings report that it must come from, and never
imports torch.

Exit codes: 0 success; 1 a report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, refused outputs, and runs that failed
a check, hit a device error or exceeded their budget; 3 this host cannot run
Metal kernels through PyTorch (not macOS, no torch, no MPS device or no
``torch.mps.compile_shader``), so ``run`` stopped before it measured or
wrote anything. These are the codes of the Day 73 to 77 Metal labs, so
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
from .memory_access_report_io import paths_alias
from .metal_fused_ceiling import load_ceiling
from .metal_fused_config import load_config
from .metal_fused_io import load_report, save_report
from .metal_fused_report import measure_report
from .metal_fused_runtime import MetalFusedUnavailableError
from .metal_fused_source import MetalFusedError
from .metal_fused_verify import MetalFusedVerificationError

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-metal-fused"
CLAIMS = "cuda=false nvidia=false triton_gpu=false hardware_counters=false"


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Check and time an unfused and three fused float32 row softmax "
            "kernels, U0 to F3, and torch.softmax on this host's Apple GPU "
            "against a measured bandwidth ceiling; no CUDA kernel, NVIDIA GPU "
            "or Triton GPU kernel is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="check and time every softmax step on this Apple GPU",
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


def short_name(name: str) -> str:
    """Spell a side as the CLI prints it: ``u0``, ``f3`` or ``torch``."""

    if name == "torch_softmax":
        return "torch"
    return name.removeprefix("softmax_").split("_")[0]


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def _flag(value: bool | None) -> str:
    return "none" if value is None else "true" if value else "false"


def _ratio(step: dict[str, object] | None) -> str:
    return "none" if step is None else f"{step['ratio']:.3f}:{step['verdict']}"


def _shape(shape: dict[str, object]) -> str:
    if "cols" in shape:
        return f"rows={shape['rows']} cols={shape['cols']}"
    return f"rows={shape['rows']} max_length={shape['max_length']}"


def plan_lines(label: str, plan: dict[str, object]) -> list[str]:
    """Print the plan that a report sealed: one line for each step."""

    return [
        f"{label} plan step={step['step']} kernel={short_name(step['variant'])} "
        f"base={short_name(step['base'])} predicted={'|'.join(step['predicted'])} "
        f"same_bits_as={short_name(step['same_bits_as'] or 'none')}"
        for step in plan["steps"]
    ]


def _ceiling_line(label: str, ceiling: dict[str, object]) -> str:
    return (
        f"{label} ceiling gb_per_second={ceiling['gb_per_second']:.2f} "
        f"kernel={ceiling['kernel']} buffer_bytes={ceiling['buffer_bytes']} "
        f"report={ceiling['report_sha256'][:12]} "
        f"measured={_flag(ceiling['measured'])} "
        f"load_before={_loads(ceiling['load_average_before'])}"
    )


def _probe_line(label: str, probe: dict[str, object]) -> str:
    return (
        f"{label} exp_probe max_ulps={probe['max_ulps']:.3f} "
        f"normal_results={probe['normal_results']} "
        f"tiny_results={probe['tiny_results']} tiny_nonzero={probe['tiny_nonzero']} "
        f"holds={_flag(probe['holds'])}"
    )


def _case_line(label: str, case: dict[str, object]) -> str:
    layout = case["layout"]
    outputs = " ".join(
        f"{short_name(output['kernel'])}_ratio={output['max_bound_ratio']:.3f} "
        f"{short_name(output['kernel'])}_repeats="
        f"{_flag(output['identical_repeats'])}"
        for output in case["outputs"]
    )
    same = " ".join(
        f"{short_name(entry['variant'])}_bits_as_{short_name(entry['as'])}="
        f"{_flag(entry['same'])}"
        for entry in case["same_bits"]
    )
    return (
        f"{label} correctness case={case['index']} rows={layout['rows']} "
        f"shortest={layout['shortest']} longest={layout['longest']} "
        f"group={layout['group_size']} inputs={case['case']['inputs']} "
        f"naive_finite={_flag(case['naive_float32_finite'])} {outputs} {same}"
    ).rstrip()


def _point_line(label: str, point: dict[str, object]) -> str:
    sides = " ".join(
        f"{short_name(name)}_gbps="
        f"{point['sides'][name]['effective_gb_per_second']:.1f} "
        f"{short_name(name)}_iqr="
        + "-".join(
            f"{rate:.1f}"
            for rate in point["sides"][name]["iqr_effective_gb_per_second"]
        )
        for name in point["side_order"]
    )
    calls = point["sides"][point["side_order"][0]]["calls_per_sample"]
    return (
        f"{label} benchmark {_shape(point['shape'])} "
        f"group={point['layout']['group_size']} calls_per_sample={calls} {sides}"
    )


def _speedup_lines(label: str, entry: dict[str, object]) -> list[str]:
    shape = _shape(entry["shape"])
    lines = []
    for row in entry["rows"]:
        rate = row["rate"]
        lines.append(
            f"{label} speedup {shape} step={row['step']} "
            f"kernel={short_name(row['variant'])} "
            + (
                "gbps=none"
                if rate is None
                else f"gbps={rate['effective_gb_per_second']:.1f} "
                f"ceiling_fraction={rate['modeled_fraction_of_ceiling']:.3f}"
            )
            + f" to_base={_ratio(row['to_base'])} to_u0={_ratio(row['to_u0'])} "
            f"to_torch={_ratio(row['to_torch'])} held={_flag(row['held'])}"
        )
    best = entry["best"]
    torch = entry["torch"]
    lines.append(
        f"{label} best {shape} kernel="
        + ("none" if best is None else short_name(best["variant"]))
        + f" to_torch={_ratio(None if best is None else best['to_torch'])} "
        f"torch_gbps={torch['rate']['effective_gb_per_second']:.1f} "
        f"torch_to_u0={_ratio(torch['to_u0'])}"
    )
    return lines


def _print_run(report: dict[str, object]) -> None:
    label = _label(report)
    load = report["host_load"]
    print(
        f"{label} load_average before={_loads(load['before'])} "
        f"after={_loads(load['after'])}"
    )
    print(_ceiling_line(label, report["ceiling"]))
    print(_probe_line(label, report["exp_probe"]))
    for refusal in report["refusals"]:
        print(
            f"{label} refused step={short_name(refusal['variant'])} "
            f"reason={refusal['reason']} "
            f"required_threads={refusal['required_threads']}"
        )
    for case in report["correctness"]["cases"]:
        print(_case_line(label, case))
    for point in report["benchmark"]["points"]:
        print(_point_line(label, point))
    for entry in report["speedup"]["points"]:
        for line in _speedup_lines(label, entry):
            print(line)
    captured = report["warnings"]
    print(
        f"{label} warnings count={captured['count']} distinct={captured['distinct']} "
        "torch_imported_before_capture="
        f"{_flag(captured['torch_imported_before_capture'])}"
    )


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    refused = ",".join(short_name(name) for name in summary["refused_steps"]) or "none"
    held = sum(record["held"] for record in summary["hypotheses"].values())
    judged = sum(
        record["held"] + record["missed"] for record in summary["hypotheses"].values()
    )
    counts = (
        f"correctness_cases={summary['correctness_cases']} "
        f"benchmark_points={summary['benchmark_points']} refused={refused} "
        f"max_bound_ratio={summary['max_bound_ratio']:.3f} "
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
        raise MetalFusedError("report output must not alias the ceiling report")


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
            ceiling = load_ceiling(args.ceilings) if args.ceilings is not None else None
            _, summary = load_report(args.report, config=config, ceiling=ceiling)
            print(_summary_line("verified_metal_fused", summary))
            return EXIT_OK
        _refuse_inputs(args.output, args.config, args.ceilings)
        config = load_config(args.config)
        ceiling = load_ceiling(args.ceilings)
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
    except MetalFusedUnavailableError as error:
        print(f"metal-fused {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except MetalFusedVerificationError as error:
        print(f"metal-fused {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"metal-fused {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in plan_lines(_label(report), report["plan"]):
        print(line)
    _print_run(report)
    print(_summary_line("metal_fused", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
