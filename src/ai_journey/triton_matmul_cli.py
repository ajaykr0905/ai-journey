"""Count, check and verify the Day 79 Triton matmul port from one command line.

``loc`` prints the line counts of the Triton text and of each Day 76 and
Day 77 kernel text under the declared rule. It needs no triton.

``check`` loads a plan, runs each case in the Triton CPU interpreter, builds
the sealed comparison report, verifies it and publishes it atomically, never
over the plan. It needs triton and torch. Triton has no macOS wheel, so on
the development host ``check`` stops with exit code 3 before it makes the
output directory or writes a byte. A backend, clock or load average injected
through ``main`` gives a report that prints and verifies as synthetic.

``verify`` recomputes a report on the CPU, optionally against the plan that
it must come from. It never imports triton or torch.

Exit codes: 0 success; 1 a report was read or built but failed verification;
2 usage errors, unreadable or invalid inputs, refused outputs and checks that
broke a rule; 3 this host cannot run the Triton CPU interpreter (no triton, no
torch, or ``TRITON_INTERPRET`` turns the interpreter off), so ``check`` ran
and wrote nothing. These are the codes of the Day 73 to 77 labs.
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
from .triton_matmul_config import load_plan
from .triton_matmul_loc import counted_texts
from .triton_matmul_report import (
    DIGEST_FIELD,
    INTERPRETER_LABEL,
    build_report,
    seal_report,
)
from .triton_matmul_runtime import InterpreterBackend, TritonMatmulUnavailableError
from .triton_matmul_verify import (
    TritonMatmulVerificationError,
    load_report,
    verify_report,
)

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
PROG = "ai-journey-triton-matmul"
LANGUAGE_TOKENS = {
    "Triton (Python)": "triton",
    "CUDA C++": "cuda",
    "Metal Shading Language": "metal",
}


def build_parser() -> argparse.ArgumentParser:
    """Define the ``loc``, ``check`` and ``verify`` commands, no abbreviations."""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Port the Day 76 tiled matmul to Triton: count its lines against the "
            "CUDA and Metal texts, check it in the Triton CPU interpreter, and "
            "verify the sealed report. No GPU runs, and nothing is timed."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "loc",
        help="print the kernel and host lines of each text",
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


def loc_lines() -> list[str]:
    """Return one line for each counted text, the Triton text first."""

    lines = []
    for text in counted_texts():
        count = text.count()
        host = "none" if count.host is None else str(count.host)
        lines.append(
            f"loc lab={text.lab} language={LANGUAGE_TOKENS[text.language]} "
            f"text={text.name} kernel={count.kernel} host={host} "
            f"other={count.other} comment={count.comment} blank={count.blank} "
            f"sha256={text.sha256}"
        )
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
    """Return one line for each case: its bound, worst ratio and exact runs."""

    lines = []
    for case in report["correctness"]["cases"]:
        shape = case["shape"]
        normal = [run for run in case["runs"] if run["family"] == "normal"]
        exact = [run for run in case["runs"] if run["family"] == "exact"]
        held = sum(run["rounded_reference_differences"] == 0 for run in exact)
        lines.append(
            f"{_label(report)} correctness m={shape['m']} n={shape['n']} "
            f"k={shape['k']} bound={case['error_bound']:.3g} configs={len(normal)} "
            f"normal_max_ratio={max(run['bound_ratio'] for run in normal):.3f} "
            f"exact_bits={held}/{len(exact)}"
        )
    return lines


def summary_line(prefix: str, report: Mapping[str, object]) -> str:
    """Return the one line that names a verified report and its counts."""

    summary = report["summary"]
    environment = report["environment"]
    load = report["host_load"]
    kind = "interpreter" if report["label"] == INTERPRETER_LABEL else "synthetic"
    runs = summary["runs"] // 2
    return (
        f"{prefix}_{kind}={report[DIGEST_FIELD]} plan={report['plan']['name']} "
        f"cases={summary['cases']} configs={summary['configs']} "
        f"within_bound={summary['normal_runs_within_bound']}/{runs} "
        f"exact_bits={summary['exact_runs_with_exact_bits']}/{runs} "
        f"max_bound_ratio={summary['max_bound_ratio']:.3f} "
        f"triton={environment['triton']} torch={environment['torch']} "
        f"machine={environment['machine']} load_before={_loads(load['before'])} "
        f"load_after={_loads(load['after'])} "
        f"kernel_lines={summary['triton_kernel_lines']} "
        f"host_lines={summary['triton_host_lines']} "
        "gpu=false autotuned=false timing=false"
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
    """

    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "loc":
            lines = loc_lines()
        elif arguments.command == "verify":
            config = arguments.config
            plan = None if config is None else load_plan(config)
            report = load_report(arguments.report, plan=plan)
            lines = [summary_line("verified_triton_matmul", report)]
        else:
            report = _check(
                arguments,
                backend,
                clock or time.time,
                load_average or os.getloadavg,
            )
            lines = [*case_lines(report), summary_line("triton_matmul", report)]
    except TritonMatmulUnavailableError as error:
        print(
            f"triton-matmul {arguments.command} unavailable: {error}", file=sys.stderr
        )
        return EXIT_UNAVAILABLE
    except TritonMatmulVerificationError as error:
        print(f"triton-matmul {arguments.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"triton-matmul {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in lines:
        print(line)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
