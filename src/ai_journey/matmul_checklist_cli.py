"""Run and verify the Day 77 performance checklist on this host's Apple GPU.

``run`` loads a config, prints the declared checklist, compiles the Day 74
naive source and the checklist library through ``torch.mps.compile_shader``,
checks every admitted kernel on the correctness cases, times every kernel
with ``torch.matmul`` on the MPS device in one rotated run, verifies the
sealed report and publishes it atomically without ever writing over the
config. A backend or clock injected through ``main`` yields a report printed
and verified as synthetic timing. ``verify`` recomputes a report on the CPU,
optionally against the config it must come from, and never imports torch.

Exit codes: 0 success; 1 a report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, refused outputs, and runs that failed
a check, hit a device error or exceeded their budget; 3 this host cannot run
Metal kernels through PyTorch (not macOS, no torch, no MPS device or no
``torch.mps.compile_shader``), so ``run`` stopped before it measured or
wrote anything. These are the codes of the Day 73 to 76 Metal labs, so
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
from .matmul_checklist_config import load_config
from .matmul_checklist_io import load_report, save_report
from .matmul_checklist_report import measure_report
from .matmul_checklist_runtime import MatmulChecklistUnavailableError
from .matmul_checklist_verify import MatmulChecklistVerificationError

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-matmul-checklist"


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Apply the Day 77 performance checklist to a float32 Metal matmul: "
            "check and time every step's kernel and torch.matmul on this host's "
            "Apple GPU; no CUDA kernel, NVIDIA GPU or cuBLAS is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="check and time every checklist kernel on this Apple GPU",
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
    """Spell a side as the CLI prints it: ``x_columns``, ``s5_pad4`` or ``torch``."""

    return "torch" if name == "torch_matmul" else name.removeprefix("matmul_")


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def _flag(value: bool | None) -> str:
    return "none" if value is None else "true" if value else "false"


def _ratio(step: dict[str, object] | None) -> str:
    return "none" if step is None else f"{step['ratio']:.3f}:{step['verdict']}"


def checklist_lines(label: str, checklist: dict[str, object]) -> list[str]:
    """Print the checklist that a report sealed: one line for each step."""

    return [
        f"{label} checklist step={step['step']} item={step['item']} "
        f"status={step['status']} base={short_name(step['base'])} "
        + " ".join(
            f"new={short_name(entry['variant'])}:{'|'.join(entry['verdicts'])}"
            for entry in step["predictions"]
        )
        for step in checklist["steps"]
    ]


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


def _speedup_lines(label: str, entry: dict[str, object]) -> list[str]:
    shape = f"m={entry['m']} n={entry['n']} k={entry['k']}"
    lines = []
    for row in entry["rows"]:
        rate = (
            "none"
            if row["rate"] is None
            else f"{row['rate']['median_gflops_per_second']:.1f}"
        )
        fraction = (
            "none"
            if row["fraction_of_torch"] is None
            else f"{row['fraction_of_torch']:.3f}"
        )
        lines.append(
            f"{label} speedup {shape} step={row['step']} "
            f"kernel={short_name(row['variant'])} gflops={rate} "
            f"to_base={_ratio(row['to_base'])} to_s0={_ratio(row['to_s0'])} "
            f"torch_fraction={fraction} held={_flag(row['held'])}"
        )
    best = entry["best"]
    lines.append(
        f"{label} best {shape} kernel={short_name(best['variant'])} "
        f"to_s0={_ratio(best['to_s0'])} "
        f"torch_fraction={best['fraction_of_torch']:.3f} "
        f"torch_gflops={entry['torch']['median_gflops_per_second']:.1f} "
        "cublas=not_measured"
    )
    return lines


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
    refused = ",".join(summary["refused_variants"]) or "none"
    held = sum(record["held"] for record in summary["hypotheses"].values())
    judged = sum(
        record["held"] + record["missed"] for record in summary["hypotheses"].values()
    )
    counts = (
        f"correctness_cases={summary['correctness_cases']} "
        f"benchmark_points={summary['benchmark_points']} refused={refused} "
        f"hypotheses_held={held}/{judged} "
        f"checklist={summary['checklist_sha256'][:12]} "
        f"scope={summary['scope']} "
        f"load_before={_loads(summary['load_average_before'])} "
        f"load_after={_loads(summary['load_average_after'])}"
    )
    claims = "cuda=false nvidia=false cublas=false hardware_counters=false"
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
            print(_summary_line("verified_matmul_checklist", summary))
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
    except MatmulChecklistUnavailableError as error:
        print(f"matmul-checklist {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except MatmulChecklistVerificationError as error:
        print(f"matmul-checklist {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"matmul-checklist {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in checklist_lines(_label(report), report["checklist"]):
        print(line)
    _print_run(report)
    print(_summary_line("matmul_checklist", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
