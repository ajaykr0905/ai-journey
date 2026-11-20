"""Write the Day 77 checklist CUDA texts, and check them without a compiler.

``sources`` writes the five kernel texts of ``gpu_checklist_cuda`` and the
Day 76 cuBLAS call text, each as its own ``.cu`` file, and a ``SHA256SUMS``
file that ``sha256sum -c`` reads on the NVIDIA host. ``check`` reads each of
those files from a directory and checks it for the 18 shapes of the Day 74
config: the nine correctness cases and the nine benchmark points. A kernel
file gets the checks of ``gpu_checklist_check``, and the cuBLAS file gets the
Day 76 checks. Each command prints one line for each source file, and
``check`` also says whether a file still has its pinned digest.

No command compiles, links or runs CUDA or cuBLAS, and none needs a GPU or
a clock, so the lab has no exit code for a missing device.

Exit codes: 0 success; 1 a text was read but failed a check; 2 usage errors,
unreadable or invalid inputs and refused outputs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .gpu_checklist_check import CHECK_SHAPES, check_cuda_source
from .gpu_checklist_cuda import (
    PINNED_KERNELS,
    GpuChecklistError,
    source_files,
    source_sha256,
)
from .gpu_tiled_cuda import check_cublas_source
from .kernel_launch_io import read_bounded, write_atomic

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
PROG = "ai-journey-gpu-checklist"
SUMS_FILE = "SHA256SUMS"
MAX_SOURCE_BYTES = 64 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Thread-coarsened and 2D register-blocked CUDA matmul texts for a "
            "later NVIDIA run, with a compiler-free check. No GPU runs, and no "
            "CUDA or cuBLAS text is compiled or called."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sources = commands.add_parser(
        "sources",
        help="write the CUDA and cuBLAS texts and their SHA256SUMS file",
        allow_abbrev=False,
    )
    sources.add_argument("--output-dir", type=Path, required=True)
    check = commands.add_parser(
        "check",
        help="check the texts in a directory for the Day 74 shapes",
        allow_abbrev=False,
    )
    check.add_argument("--source-dir", type=Path, required=True)
    return parser


def write_sources(directory: Path) -> list[str]:
    """Write each pinned text and the ``SHA256SUMS`` file; return the lines."""

    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise GpuChecklistError("output directory must be a directory, not a symlink")
    lines, sums = [], []
    for name, text in source_files():
        digest = source_sha256(text)
        write_atomic(directory / name, text.encode("ascii"))
        sums.append(f"{digest}  {name}\n")
        lines.append(f"cuda_source={name} sha256={digest} compiled=false")
    write_atomic(directory / SUMS_FILE, "".join(sums).encode("ascii"))
    return lines


def _read_text(path: Path) -> str:
    payload = read_bounded(path, limit=MAX_SOURCE_BYTES, label=path.name)
    try:
        return payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise GpuChecklistError(f"{path.name} must be ASCII text") from error


def check_sources(directory: Path) -> tuple[list[str], list[str]]:
    """Check each pinned file in ``directory``; return its lines and failures.

    A file that is missing or unreadable raises, because no check ran on it.
    """

    kernels = {f"{kernel.name}.cu": kernel for kernel in PINNED_KERNELS}
    lines, failures = [], []
    for name, pinned in source_files():
        text = _read_text(directory / name)
        results = [
            check_cuda_source(text, kernel=kernels[name], shape=shape)
            if name in kernels
            else check_cublas_source(text, shape=shape)
            for shape in CHECK_SHAPES
        ]
        passed = sum(result for item in results for _, result in item.checks)
        total = sum(len(item.checks) for item in results)
        failed = dict.fromkeys(failure for item in results for failure in item.failures)
        if failed:
            failures.append(f"{name} fails {', '.join(failed)}")
        digest = source_sha256(text)
        lines.append(
            f"cuda_check={name} sha256={digest} "
            f"pinned={str(digest == source_sha256(pinned)).lower()} "
            f"checks={passed}/{total} shapes={len(CHECK_SHAPES)} compiled=false"
        )
    return lines, failures


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its explicit exit code."""

    arguments = build_parser().parse_args(argv)
    failures: list[str] = []
    try:
        if arguments.command == "sources":
            lines = write_sources(arguments.output_dir)
        else:
            lines, failures = check_sources(arguments.source_dir)
    except (OSError, TypeError, ValueError) as error:
        print(f"gpu-checklist {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in lines:
        print(line)
    for failure in failures:
        print(f"gpu-checklist check rejected: {failure}", file=sys.stderr)
    return EXIT_REJECTED if failures else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
