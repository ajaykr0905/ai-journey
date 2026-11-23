"""Model, check and verify the Day 81 flash attention lab from one command line.

``model`` prints the declared memory and traffic table of naive and flash
attention, per head, for N = 512 to 8192 and d = 64 and 128. It is a model of
allocated intermediates and bytes moved, not a measured GPU footprint. It
needs no triton.

``check`` loads a plan, runs each case in the Triton CPU interpreter, builds
the sealed report, verifies it and publishes it atomically, never over the
plan. It needs triton and torch. Triton has no macOS wheel, so on the
development host ``check`` stops with exit code 3 before it makes the output
directory or writes a byte. A backend injected through ``main`` gives a report
that prints and verifies as synthetic.

``verify`` recomputes a report on the CPU, optionally against the plan that it
must come from. It never imports triton or torch.

Exit codes: 0 success; 1 a report was read or built but failed verification;
2 usage errors, unreadable or invalid inputs, refused outputs and checks that
broke a rule; 3 this host cannot run the Triton CPU interpreter, so ``check``
ran and wrote nothing. These are the codes of the Day 73 to 80 labs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .kernel_launch_io import canonical_json_bytes, write_atomic
from .kernel_metal_cli import _refuse_unusable_output
from .triton_flash_config import load_plan
from .triton_flash_memory import MEMORY_TABLE, STATEMENT, mebibytes
from .triton_flash_report import (
    DIGEST_FIELD,
    INTERPRETER_LABEL,
    build_report,
    seal_report,
)
from .triton_flash_runtime import InterpreterBackend, TritonFlashUnavailableError
from .triton_flash_verify import (
    TritonFlashVerificationError,
    load_report,
    verify_report,
)

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-triton-flash"


def build_parser() -> argparse.ArgumentParser:
    """Define the ``model``, ``check`` and ``verify`` commands, no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Model the memory of naive and flash attention, check a Triton flash "
            "attention forward pass in the Triton CPU interpreter, and verify the "
            "sealed report. No GPU runs, and nothing is timed."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "model",
        help="print the declared memory and traffic table per head",
        allow_abbrev=False,
    )
    check = commands.add_parser(
        "check",
        help="run a plan in the Triton CPU interpreter and write a sealed report",
        allow_abbrev=False,
    )
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify",
        help="recompute a sealed report on the CPU, without triton",
        allow_abbrev=False,
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    return parser


def model_lines() -> list[str]:
    """Return one line for each row of the model, then the boundary line."""

    lines = []
    for row in MEMORY_TABLE:
        item = row.to_mapping()
        lines.append(
            f"memory n={row.sequence} d={row.head_dim} "
            f"naive_bytes={item['naive_bytes']} flash_bytes={item['flash_bytes']} "
            f"naive_mib={mebibytes(item['naive_bytes'])} "
            f"flash_mib={mebibytes(item['flash_bytes'])} "
            f"memory_ratio={item['memory_ratio']:.2f} "
            f"naive_traffic_mib={mebibytes(item['naive_traffic_bytes'])} "
            f"flash_traffic_mib={mebibytes(item['flash_traffic_bytes'])} "
            f"traffic_ratio={item['traffic_ratio']:.2f}"
        )
    lines.append(f"model=declared measured=false gpu=false note={STATEMENT!r}")
    return lines


def _label(report: Mapping[str, object]) -> str:
    return (
        "triton_interpreter_check"
        if report["label"] == INTERPRETER_LABEL
        else "synthetic_backend"
    )


def _loads(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def case_lines(report: Mapping[str, object]) -> list[str]:
    """Return one line for each case: its shape and its worst ratios."""

    lines = []
    for case in report["correctness"]["cases"]:
        shape = case["case"]
        runs = case["runs"]
        lines.append(
            f"{_label(report)} correctness bh={shape['batch_heads']} "
            f"n={shape['sequence']} d={shape['head_dim']} "
            f"causal={str(shape['causal']).lower()} runs={len(runs)} "
            f"max_output_ratio={max(r['output_bound_ratio'] for r in runs):.4f} "
            f"max_lse_ratio={max(r['lse_bound_ratio'] for r in runs):.4f}"
        )
    return lines


def summary_line(prefix: str, report: Mapping[str, object]) -> str:
    """Return the one line that names a verified report and its counts."""

    summary = report["summary"]
    environment = report["environment"]
    load = report["host_load"]
    kind = "interpreter" if report["label"] == INTERPRETER_LABEL else "synthetic"
    return (
        f"{prefix}_{kind}={report[DIGEST_FIELD]} plan={report['plan']['name']} "
        f"cases={summary['cases']} configs={summary['configs']} "
        f"within_bound={summary['runs_within_bound']}/{summary['runs']} "
        f"max_output_ratio={summary['max_output_bound_ratio']:.4f} "
        f"max_lse_ratio={summary['max_lse_bound_ratio']:.4f} "
        f"triton={environment['triton']} torch={environment['torch']} "
        f"machine={environment['machine']} load_before={_loads(load['before'])} "
        f"load_after={_loads(load['after'])} "
        "gpu=false autotuned=false timing=false memory_measured=false"
    )


def _check(
    arguments: argparse.Namespace,
    backend: object | None,
    clock: Callable[[], float],
    load_average: Callable[[], tuple[float, float, float]],
) -> dict[str, object]:
    # Refuse a bad output and a bad plan before the interpreter runs.
    _refuse_unusable_output(arguments.output, arguments.config)
    plan = load_plan(arguments.config)
    chosen = InterpreterBackend() if backend is None else backend
    report = seal_report(
        build_report(plan, chosen, clock=clock, load_average=load_average)
    )
    verify_report(report, plan=plan)
    write_atomic(
        arguments.output,
        canonical_json_bytes(report) + b"\n",
        protected=(arguments.config,),
    )
    return report


def main(
    argv: Sequence[str] | None = None,
    *,
    backend: object | None = None,
    clock: Callable[[], float] | None = None,
    load_average: Callable[[], tuple[float, float, float]] | None = None,
) -> int:
    """Run one command and return its exit code; the keywords are for tests.

    A replaced backend makes the report synthetic, never an interpreter check.
    The interpreter backend refuses a replaced clock or load average.
    """

    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "model":
            lines = model_lines()
        elif arguments.command == "verify":
            config = arguments.config
            plan = None if config is None else load_plan(config)
            report = load_report(arguments.report, plan=plan)
            lines = [summary_line("verified_triton_flash", report)]
        else:
            report = _check(
                arguments,
                backend,
                clock or time.time,
                load_average or os.getloadavg,
            )
            lines = [*case_lines(report), summary_line("triton_flash", report)]
    except TritonFlashUnavailableError as error:
        print(f"triton-flash {arguments.command} unavailable: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except TritonFlashVerificationError as error:
        print(f"triton-flash {arguments.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"triton-flash {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in lines:
        print(line)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
