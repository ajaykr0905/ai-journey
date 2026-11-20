"""Run and verify the Day 78 sum reductions on this host's Apple GPU.

``run`` loads a config and a gpu-ceilings report, which the Day 74 verifier
must accept, prints the declared plan, compiles the reduction library
through ``torch.mps.compile_shader``, checks every kernel on the
correctness cases, times every kernel with ``torch.sum`` on the MPS device
in one rotated run, verifies the sealed report and publishes it atomically
without ever writing over its inputs. A backend or clock injected through
``main`` yields a report printed and verified as synthetic timing.
``verify`` recomputes a report on the CPU, optionally against the config
and the gpu-ceilings report it must come from, and never imports torch.

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

from .gpu_reduce_ceiling import load_ceiling
from .gpu_reduce_config import load_config
from .gpu_reduce_io import load_report, save_report
from .gpu_reduce_report import measure_report
from .gpu_reduce_runtime import GpuReduceUnavailableError
from .gpu_reduce_source import GpuReduceError
from .gpu_reduce_verify import GpuReduceVerificationError
from .kernel_metal_cli import _refuse_unusable_output
from .memory_access_report_io import paths_alias

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-gpu-reduce"


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Check and time six float32 sum reductions, R0 to R5, and torch.sum "
            "on this host's Apple GPU against a measured bandwidth ceiling; no "
            "CUDA kernel or NVIDIA GPU is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="check and time every reduction on this Apple GPU",
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
    """Spell a side as the CLI prints it: ``r0_interleaved`` or ``torch``."""

    return "torch" if name == "torch_sum" else name.removeprefix("reduce_")


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def _flag(value: bool | None) -> str:
    return "none" if value is None else "true" if value else "false"


def _ratio(step: dict[str, object] | None) -> str:
    return "none" if step is None else f"{step['ratio']:.3f}:{step['verdict']}"


def _indices(values: list[int]) -> str:
    return ",".join(map(str, values)) or "none"


def plan_lines(label: str, plan: dict[str, object]) -> list[str]:
    """Print the plan that a report sealed: one line for each step."""

    return [
        f"{label} plan step={step['step']} kernel={short_name(step['kernel'])} "
        f"base={short_name(step['base'])} "
        + " ".join(
            f"{entry['size_class']}={'|'.join(entry['verdicts'])}"
            for entry in step["predictions"]
        )
        for step in plan["steps"]
    ]


def _ceiling_line(label: str, ceiling: dict[str, object]) -> str:
    return (
        f"{label} ceiling gb_per_second={ceiling['gb_per_second']:.2f} "
        f"kernel={ceiling['kernel']} buffer_bytes={ceiling['buffer_bytes']} "
        f"measured={_flag(ceiling['measured'])} "
        f"same_environment={_flag(ceiling['same_environment'])} "
        f"report={ceiling['report_sha256'][:12]} "
        f"load_before={_loads(ceiling['load_average_before'])} "
        f"load_after={_loads(ceiling['load_average_after'])}"
    )


def _case_line(label: str, case: dict[str, object]) -> str:
    outputs = " ".join(
        f"{short_name(output['kernel'])}={output['bound_ratio']:.3g}:"
        f"{'same' if output['same_bits_every_run'] else 'changed'}"
        for output in case["outputs"]
    )
    return (
        f"{label} correctness pattern={case['pattern']} n={case['n']} "
        f"exact={_flag(case['exact_in_any_order'])} {outputs}"
    )


def _point_line(label: str, point: dict[str, object]) -> str:
    sides = " ".join(
        f"{short_name(side)}_gbs={point['sides'][side]['median_gb_per_second']:.1f} "
        f"{short_name(side)}_iqr="
        + "-".join(f"{rate:.1f}" for rate in point["sides"][side]["iqr_gb_per_second"])
        for side in point["side_order"]
    )
    return f"{label} benchmark n={point['n']} {sides}"


def _speedup_lines(label: str, entry: dict[str, object]) -> list[str]:
    lines = []
    for row in entry["rows"]:
        lines.append(
            f"{label} speedup n={entry['n']} step={row['step']} "
            f"kernel={short_name(row['kernel'])} "
            f"gbs={row['rate']['median_gb_per_second']:.1f} "
            f"ceiling_fraction={row['fraction_of_ceiling']:.3f} "
            f"torch_fraction={row['fraction_of_torch_sum']:.3f} "
            f"to_base={_ratio(row['to_base'])} to_r0={_ratio(row['to_r0'])} "
            f"held={_flag(row['held'])}"
        )
    best = entry["best"]
    lines.append(
        f"{label} best n={entry['n']} kernel={short_name(best['kernel'])} "
        f"gbs={best['median_gb_per_second']:.1f} "
        f"ceiling_fraction={best['fraction_of_ceiling']:.3f} "
        f"to_r0={_ratio(best['to_r0'])} "
        f"torch_gbs={entry['torch_sum']['median_gb_per_second']:.1f} "
        f"torch_ceiling_fraction={entry['torch_sum']['fraction_of_ceiling']:.3f}"
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
    for case in report["correctness"]["cases"]:
        print(_case_line(label, case))
    for point in report["benchmark"]["points"]:
        print(_point_line(label, point))
    for entry in report["speedup"]["points"]:
        for line in _speedup_lines(label, entry):
            print(line)
    for record in report["summary"]["same_order"]:
        print(
            f"{label} same_order kernel={short_name(record['kernel'])} "
            f"as={short_name(record['same_order_as'])} "
            f"cases_with_other_bits={_indices(record['cases_with_other_bits'])} "
            f"points_with_other_bits={_indices(record['points_with_other_bits'])}"
        )
    captured = report["warnings"]
    print(
        f"{label} warnings count={captured['count']} distinct={captured['distinct']} "
        "torch_imported_before_capture="
        f"{_flag(captured['torch_imported_before_capture'])}"
    )


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    held = sum(record["held"] for record in summary["hypotheses"].values())
    judged = held + sum(record["missed"] for record in summary["hypotheses"].values())
    counts = (
        f"correctness_cases={summary['correctness_cases']} "
        f"benchmark_points={summary['benchmark_points']} "
        f"hypotheses_held={held}/{judged} "
        f"plan={summary['plan_sha256'][:12]} "
        f"ceiling={summary['ceiling_report_sha256'][:12]} "
        f"fraction_measured={_flag(summary['fraction_measured'])} "
        f"scope={summary['scope']} "
        f"load_before={_loads(summary['load_average_before'])} "
        f"load_after={_loads(summary['load_average_after'])}"
    )
    claims = "cuda=false nvidia=false warp_level=false hardware_counters=false"
    if summary["measured"]:
        return f"{prefix}_measurement={summary['report_sha256']} {counts} {claims}"
    return (
        f"{prefix}_synthetic_timing={summary['report_sha256']} {counts} "
        f"measured=false {claims}"
    )


def _refuse_inputs(args: argparse.Namespace) -> None:
    _refuse_unusable_output(args.output, args.config)
    if paths_alias(args.output, args.ceilings):
        raise GpuReduceError("report output must not alias the ceilings report")


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
            print(_summary_line("verified_gpu_reduce", summary))
            return EXIT_OK
        _refuse_inputs(args)
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
    except GpuReduceUnavailableError as error:
        print(f"gpu-reduce {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except GpuReduceVerificationError as error:
        print(f"gpu-reduce {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"gpu-reduce {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in plan_lines(_label(report), report["plan"]):
        print(line)
    _print_run(report)
    print(_summary_line("gpu_reduce", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
