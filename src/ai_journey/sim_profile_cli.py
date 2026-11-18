"""Profile the pure-Python launch simulator, seal the rounds, verify reports.

``run`` profiles and times the current tree at one stage of the config and
writes a sealed snapshot. ``rounds`` reads one snapshot per stage, in stage
order, verifies each one, and seals the round-by-round comparison.
``verify`` checks a snapshot or a rounds report; with ``--config`` it also
binds the report to a config file. No command uses a GPU or imports torch or
NumPy. The timings are host-specific CPU measurements.

Exit codes: 0 success; 1 a report was read or built but failed its checks,
or the simulator's misuse records changed; 2 usage errors, unreadable or
invalid inputs, and refused outputs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .kernel_launch_io import paths_alias
from .sim_profile_config import (
    SimProfileError,
    SimProfileVerificationError,
    load_config,
)
from .sim_profile_io import load_report, read_report, save_report
from .sim_profile_rounds import rounds_report
from .sim_profile_semantics import SEMANTICS_SHA256
from .sim_profile_snapshot import snapshot

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
PROG = "ai-journey-sim-profile"
_BOUNDARY = "gpu_executed=false cuda_compiled=false hardware_counters=false"


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run``, ``rounds`` and ``verify`` commands, no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Profile the pure-Python Day 73 launch simulator with cProfile and "
            "time it with perf_counter_ns; no GPU runs."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="profile and time the current tree at one stage; write a snapshot",
        allow_abbrev=False,
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--stage", required=True)
    run.add_argument("--output", type=Path, required=True)
    rounds = commands.add_parser(
        "rounds",
        help="verify one snapshot per stage and seal the round comparison",
        allow_abbrev=False,
    )
    rounds.add_argument("--config", type=Path, required=True)
    rounds.add_argument("--snapshot", type=Path, action="append", required=True)
    rounds.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="check a snapshot or a rounds report", allow_abbrev=False
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    return parser


def _label(report: Mapping[str, object]) -> str:
    return "cpu_measurement" if report["measured"] else "synthetic_timing"


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def snapshot_lines(report: Mapping[str, object]) -> list[str]:
    """Return the stage line, one line per hotspot and one per workload."""

    label = _label(report)
    lines = [
        f"{label} stage={report['stage']} "
        f"load_before={_loads(report['load_average_before'])} "
        f"load_after={_loads(report['load_average_after'])} "
        f"function_calls={report['profile']['calls']}"
    ]
    for item in report["profile"]["hotspots"]:
        lines.append(
            f"{label} hotspot rank={item['rank']} function={item['function']} "
            f"calls={item['calls']} self_ns={item['total_ns']} "
            f"share_ppm={item['share_ppm']}"
        )
    for entry in report["workloads"]:
        timing = entry["timing"]["timing"]
        lines.append(
            f"{label} workload name={entry['name']} "
            f"median_ns={timing['median_ns']} iqr_ns={timing['iqr_ns']} "
            f"reads={entry['reads']} writes={entry['writes']}"
        )
    return lines


def rounds_lines(report: Mapping[str, object]) -> list[str]:
    """Return one line per stage and one line per round."""

    label = _label(report)
    lines = [
        f"{label} stage={item['stage']} "
        f"load_before={_loads(item['load_average_before'])} "
        f"load_after={_loads(item['load_average_after'])} "
        f"total_median_ns={item['total_median_ns']} "
        f"top_hotspot={item['top_hotspot']}"
        for item in report["stages"]
    ]
    for item in report["rounds"]:
        lines.append(
            f"{label} round={item['round']} target={item['target']} "
            f"rank_before={item['target_before']['rank']} "
            f"calls_before={item['target_before']['calls']} "
            f"calls_after={item['target_after']['calls']} "
            f"total_median_before_ns={item['total']['before_median_ns']} "
            f"total_median_after_ns={item['total']['after_median_ns']} "
            f"speedup={item['total']['speedup']}"
        )
    return lines


def summary_line(prefix: str, summary: Mapping[str, object]) -> str:
    """Return the one line that names a verified report and its headline."""

    if summary["kind"] == "sim_profile_snapshot":
        detail = (
            f"stage={summary['stage']} total_median_ns={summary['total_median_ns']} "
            f"top_hotspot={summary['top_hotspot']}"
        )
    else:
        speedups = ",".join(str(value) for value in summary["round_speedups"])
        detail = (
            f"rounds={summary['rounds']} round_speedups={speedups} "
            f"speedup={summary['speedup']}"
        )
    return (
        f"{prefix}={summary['report_sha256']} {detail} scope={summary['scope']} "
        f"semantics_sha256={SEMANTICS_SHA256} {_BOUNDARY}"
    )


def _refuse_alias(output: Path, inputs: Sequence[Path]) -> None:
    for item in inputs:
        if paths_alias(output, item):
            raise SimProfileError("output must not alias an input")


def _run(arguments: argparse.Namespace) -> tuple[list[str], str, dict[str, object]]:
    _refuse_alias(arguments.output, [arguments.config])
    config = load_config(arguments.config)
    report = snapshot(config, arguments.stage)
    summary = save_report(arguments.output, report, protected=(arguments.config,))
    return snapshot_lines(report), "sim_profile_snapshot", summary


def _rounds(arguments: argparse.Namespace) -> tuple[list[str], str, dict[str, object]]:
    inputs = [arguments.config, *arguments.snapshot]
    _refuse_alias(arguments.output, inputs)
    config = load_config(arguments.config)
    snapshots = [read_report(path, label="snapshot") for path in arguments.snapshot]
    report = rounds_report(config, snapshots)
    summary = save_report(arguments.output, report, protected=inputs)
    return rounds_lines(report), "sim_profile_rounds", summary


def _verify(arguments: argparse.Namespace) -> tuple[list[str], str, dict[str, object]]:
    config = None if arguments.config is None else load_config(arguments.config)
    report, summary = load_report(arguments.report, config=config)
    lines = (
        snapshot_lines(report)
        if summary["kind"] == "sim_profile_snapshot"
        else rounds_lines(report)
    )
    return lines, f"verified_{summary['kind']}", summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its explicit exit code."""

    arguments = build_parser().parse_args(argv)
    commands = {"run": _run, "rounds": _rounds, "verify": _verify}
    try:
        lines, prefix, summary = commands[arguments.command](arguments)
    except SimProfileVerificationError as error:
        print(f"sim-profile {arguments.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError) as error:
        print(f"sim-profile {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in lines:
        print(line)
    print(summary_line(prefix, summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
