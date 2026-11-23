"""Run and verify the Day 78 scan lab on this host's Apple GPU.

``run`` loads a config, prints the declared plan, compiles the scan library
and the Day 74 ceiling source through ``torch.mps.compile_shader``, checks
every admitted kernel and ``torch.cumsum`` on the correctness cases, times
every side in one rotated run, verifies the sealed report and publishes it
atomically without ever writing over the config. A backend or clock injected
through ``main`` yields a report printed and verified as synthetic timing.
``verify`` recomputes a report on the CPU, optionally against the config it
must come from, and never imports torch.

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

from .gpu_scan_config import load_config
from .gpu_scan_io import load_report, save_report
from .gpu_scan_report import measure_report
from .gpu_scan_runtime import GpuScanUnavailableError
from .gpu_scan_verify import GpuScanVerificationError
from .kernel_metal_cli import _refuse_unusable_output

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-gpu-scan"
SHORT_NAMES = {"torch_cumsum": "torch", "stream_copy": "copy"}


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Check and time three float32 Metal scans, torch.cumsum and a copy "
            "on this host's Apple GPU; no CUDA kernel or NVIDIA GPU is used."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="check and time every scan on this Apple GPU",
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
    """Spell a side as the CLI prints it: ``s2_simd``, ``torch`` or ``copy``."""

    return SHORT_NAMES.get(name, name.removeprefix("scan_"))


def _label(report: dict[str, object]) -> str:
    return "apple_gpu_metal_measurement" if report["measured"] else "synthetic_timing"


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def _flag(value: bool | None) -> str:
    return "none" if value is None else "true" if value else "false"


def _ratio(step: dict[str, object] | None) -> str:
    return "none" if step is None else f"{step['ratio']:.3f}:{step['verdict']}"


def _number(value: float | None, digits: int = 3) -> str:
    return "none" if value is None else f"{value:.{digits}f}"


def plan_lines(label: str, plan: dict[str, object]) -> list[str]:
    """Print the plan that a report sealed: one line for each hypothesis."""

    return [
        f"{label} plan step={entry['step']} kernel={short_name(entry['kernel'])} "
        f"base={short_name(entry['base'])} predictions="
        + ",".join(
            f"{item['length']}:{'|'.join(item['verdicts'])}"
            for item in entry["predictions"]
        )
        for entry in plan["hypotheses"]
    ]


def _bits(output: dict[str, object]) -> str:
    if output["exact_bits"] is not None:
        return "exact" if output["exact_bits"] else "other"
    if output["replay_bits"] is not None:
        return "replay" if output["replay_bits"] else "other"
    return "unknown"


def _case_line(label: str, case: dict[str, object]) -> str:
    sides = " ".join(
        f"{short_name(output['side'])}_error={output['max_relative_error']:.3g} "
        f"{short_name(output['side'])}_bits={_bits(output)} "
        f"{short_name(output['side'])}_deterministic={_flag(output['deterministic'])}"
        for output in case["outputs"]
    )
    levels = ",".join(str(length) for length in case["levels"])
    return (
        f"{label} correctness length={case['length']} pattern={case['pattern']} "
        f"levels={levels} {sides}"
    )


def _point_line(label: str, point: dict[str, object]) -> str:
    sides = " ".join(
        f"{short_name(side)}_gbps="
        f"{point['sides'][side]['median_gb_per_second']:.2f} "
        f"{short_name(side)}_iqr="
        + "-".join(f"{rate:.2f}" for rate in point["sides"][side]["iqr_gb_per_second"])
        for side in point["side_order"]
    )
    return (
        f"{label} benchmark length={point['length']} "
        f"dispatches={point['dispatches']} {sides}"
    )


def _speedup_lines(label: str, entry: dict[str, object]) -> list[str]:
    lines = []
    for row in entry["rows"]:
        rate = (
            "none"
            if row["rate"] is None
            else f"{row['rate']['median_gb_per_second']:.2f}"
        )
        lines.append(
            f"{label} speedup length={entry['length']} step={row['step']} "
            f"kernel={short_name(row['kernel'])} gbps={rate} "
            f"to_base={_ratio(row['to_base'])} to_s0={_ratio(row['to_s0'])} "
            f"to_torch={_ratio(row['to_vendor'])} "
            f"copy_fraction={_number(row['fraction_of_copy'])} "
            f"modeled_copy_fraction={_number(row['modeled_fraction_of_copy'])} "
            f"held={_flag(row['held'])}"
        )
    best = entry["best"]
    lines.append(
        f"{label} best length={entry['length']} "
        f"kernel={'none' if best is None else short_name(best['kernel'])} "
        f"to_torch={'none' if best is None else _ratio(best['to_vendor'])} "
        f"torch_gbps={entry['vendor']['median_gb_per_second']:.2f} "
        f"torch_copy_fraction={entry['vendor']['fraction_of_copy']:.3f} "
        f"copy_gbps={entry['copy']['median_gb_per_second']:.2f}"
    )
    return lines


def _print_run(report: dict[str, object]) -> None:
    label = _label(report)
    for line in plan_lines(label, report["plan"]):
        print(line)
    load = report["host_load"]
    print(
        f"{label} load_average before={_loads(load['before'])} "
        f"after={_loads(load['after'])}"
    )
    for refusal in report["refusals"]:
        print(
            f"{label} refused kernel={refusal['name']} reason={refusal['reason']} "
            f"allowed_threads={refusal['allowed_threads']} "
            f"static_threadgroup_bytes={refusal['static_threadgroup_bytes']} "
            f"simd_width={refusal['simd_width']}"
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
    refused = ",".join(summary["refused_kernels"]) or "none"
    held = sum(record["held"] for record in summary["hypotheses"].values())
    judged = sum(
        record["held"] + record["missed"] for record in summary["hypotheses"].values()
    )
    counts = (
        f"correctness_cases={summary['correctness_cases']} "
        f"benchmark_points={summary['benchmark_points']} refused={refused} "
        f"hypotheses_held={held}/{judged} "
        f"plan={summary['plan_sha256'][:12]} "
        f"scope={summary['scope']} "
        f"load_before={_loads(summary['load_average_before'])} "
        f"load_after={_loads(summary['load_average_after'])}"
    )
    claims = "cuda=false nvidia=false warp_shuffles=false hardware_counters=false"
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
            print(_summary_line("verified_gpu_scan", summary))
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
    except GpuScanUnavailableError as error:
        print(f"gpu-scan {args.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except GpuScanVerificationError as error:
        print(f"gpu-scan {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"gpu-scan {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    _print_run(report)
    print(_summary_line("gpu_scan", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
