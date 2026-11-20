"""Write the Day 77 checklist CUDA texts for a later NVIDIA run.

``sources`` writes the five kernel texts of ``gpu_checklist_cuda`` and the
Day 76 cuBLAS call text, each as its own ``.cu`` file, and a ``SHA256SUMS``
file that ``sha256sum -c`` reads on the NVIDIA host. It prints one line for
each source file.

No command compiles, links or runs CUDA or cuBLAS, and none needs a GPU or
a clock, so the lab has no exit code for a missing device.

Exit codes: 0 success; 2 usage errors, invalid inputs and refused outputs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .gpu_checklist_cuda import GpuChecklistError, source_files, source_sha256
from .kernel_launch_io import write_atomic

EXIT_OK = 0
EXIT_INVALID = 2
PROG = "ai-journey-gpu-checklist"
SUMS_FILE = "SHA256SUMS"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Thread-coarsened and 2D register-blocked CUDA matmul texts for a "
            "later NVIDIA run. No GPU runs, and no CUDA or cuBLAS text is "
            "compiled or called."
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its explicit exit code."""

    arguments = build_parser().parse_args(argv)
    try:
        lines = write_sources(arguments.output_dir)
    except (OSError, TypeError, ValueError) as error:
        print(f"gpu-checklist {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in lines:
        print(line)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
