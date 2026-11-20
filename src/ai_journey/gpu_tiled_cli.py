"""Run and verify the tiled CUDA contract lab, and write its source texts.

``run`` builds the report of a plan on the CPU, seals it, verifies it and
publishes it atomically, never over the plan. It uses the pinned Day 76 plan
unless ``--config`` names a plan file. ``verify`` loads a report and
recomputes every field from its embedded plan; ``--config`` also binds the
report to a plan file. ``sources`` writes the three tiled kernel texts and
the cuBLAS call text with a ``SHA256SUMS`` file, for a later NVIDIA run.
Each command prints one line for each report or file.

No command compiles, links or runs CUDA or cuBLAS, and none needs a GPU or
a clock, so the lab has no exit code for a missing device.

Exit codes: 0 success; 1 a report was read or built but failed verification;
2 usage errors, unreadable or invalid inputs, and refused outputs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .gpu_tiled_config import PINNED_PLAN, load_plan
from .gpu_tiled_cuda import (
    TILES,
    GpuTiledError,
    cublas_source,
    source_sha256,
    tiled_cuda_source,
)
from .gpu_tiled_report import DIGEST_FIELD, build_report, seal_report
from .gpu_tiled_verify import GpuTiledVerificationError, load_report, verify_report
from .kernel_launch_io import canonical_json_bytes, paths_alias, write_atomic

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
PROG = "ai-journey-gpu-tiled"
SUMS_FILE = "SHA256SUMS"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Tiled CUDA matmul and cuBLAS SGEMM source text with a compiler-free "
            "check, a CPU block simulator and a declared occupancy estimate. No "
            "GPU runs, and no CUDA or cuBLAS text is compiled or called."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="check and simulate a plan and write a sealed report",
        allow_abbrev=False,
    )
    run.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="recompute a sealed report", allow_abbrev=False
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    sources = commands.add_parser(
        "sources",
        help="write the CUDA and cuBLAS texts for a later NVIDIA run",
        allow_abbrev=False,
    )
    sources.add_argument("--output-dir", type=Path, required=True)
    return parser


def summary_line(prefix: str, report: Mapping[str, object]) -> str:
    """Return the one line that names a verified report and its headline counts."""

    summary = report["summary"]
    runs = summary["block_runs"]
    return (
        f"{prefix}={report[DIGEST_FIELD]} plan={report['plan']['name']} "
        f"cases={summary['cases']} tiles={summary['tiles']} "
        f"replay_matches={summary['block_runs_matching_replay']}/{runs} "
        f"traffic_matches={summary['block_runs_matching_traffic']}/{runs} "
        f"race_free_runs={summary['race_free_block_runs']}/{runs} "
        f"cuda_checks={summary['cuda_checks_passed']}/{summary['cuda_checks']} "
        f"cublas_checks={summary['cublas_checks_passed']}/{summary['cublas_checks']} "
        f"occupancy_estimates={summary['occupancy_estimates']} "
        "simulated_on_cpu=true gpu_executed=false cuda_compiled=false "
        "cublas_called=false"
    )


def source_files() -> tuple[tuple[str, str], ...]:
    """Return the file name and the text of each source, tiles first."""

    return (
        *((f"tiled_matmul_t{tile}.cu", tiled_cuda_source(tile=tile)) for tile in TILES),
        ("sgemm_row_major.cu", cublas_source()),
    )


def _run(arguments: argparse.Namespace) -> dict[str, object]:
    config = arguments.config
    # Refuse a taken plan path before spending time on the simulation.
    if config is not None and paths_alias(arguments.output, config):
        raise GpuTiledError("report output must not alias the plan")
    plan = PINNED_PLAN if config is None else load_plan(config)
    report = seal_report(build_report(plan))
    verify_report(report, plan=plan)
    write_atomic(
        arguments.output,
        canonical_json_bytes(report) + b"\n",
        protected=() if config is None else (config,),
    )
    return report


def _verify(arguments: argparse.Namespace) -> dict[str, object]:
    plan = None if arguments.config is None else load_plan(arguments.config)
    return load_report(arguments.report, plan=plan)


def _sources(arguments: argparse.Namespace) -> list[str]:
    directory = arguments.output_dir
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise GpuTiledError("output directory must be a directory, not a symlink")
    lines, sums = [], []
    for name, text in source_files():
        digest = source_sha256(text)
        write_atomic(directory / name, text.encode("ascii"))
        sums.append(f"{digest}  {name}\n")
        lines.append(f"cuda_source={name} sha256={digest} compiled=false")
    write_atomic(directory / SUMS_FILE, "".join(sums).encode("ascii"))
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its explicit exit code."""

    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "sources":
            lines = _sources(arguments)
        elif arguments.command == "run":
            lines = [summary_line("gpu_tiled_report", _run(arguments))]
        else:
            report = _verify(arguments)
            lines = [summary_line("verified_gpu_tiled_report", report)]
    except GpuTiledVerificationError as error:
        print(f"gpu-tiled {arguments.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError) as error:
        print(f"gpu-tiled {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in lines:
        print(line)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
