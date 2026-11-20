"""Write the Day 78 CUDA reduction texts, check them and predict their results.

``sources`` writes the two texts of ``gpu_reduce_cuda``, each as its own
``.cu`` file, and a ``SHA256SUMS`` file that ``sha256sum -c`` reads on the
NVIDIA host. ``check`` reads each of those files from a directory and checks
it with ``gpu_reduce_cuda_check`` for the 14 sizes of the Day 78 config: the
correctness sizes and the timing sizes. Each command prints one line for
each source file, and ``check`` also says whether a file still has its
pinned digest.

``simulate`` reads a Day 78 config, runs both schedules on the CPU for each
correctness case and each timing size, and prints one line for each kernel
and input, with the predicted bits, and a summary line with the digest of
all predictions. A later NVIDIA run compares its results with these bits.

No command compiles, links or runs CUDA, and none needs a GPU or a clock, so
the lab has no exit code for a missing device.

Exit codes: 0 success; 1 a text was read but failed a check, or a predicted
result broke its bound, its counts or its hazard log; 2 usage errors,
unreadable or invalid inputs and refused outputs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .gpu_reduce_cuda import (
    PINNED_KERNELS,
    GpuReduceCudaError,
    source_files,
    source_sha256,
)
from .gpu_reduce_config import load_config
from .gpu_reduce_cuda_check import CHECK_SIZES, check_reduce_source
from .gpu_reduce_cuda_sim import Prediction, predict, predictions_sha256
from .kernel_launch_io import read_bounded, write_atomic

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
PROG = "ai-journey-gpu-reduce-cuda"
SUMS_FILE = "SHA256SUMS"
MAX_SOURCE_BYTES = 64 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "CUDA texts of the R1 shared-memory and the warp-shuffle sum "
            "reductions for a later NVIDIA run. No GPU runs, and no CUDA text "
            "is compiled or run."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sources = commands.add_parser(
        "sources",
        help="write the two CUDA texts and their SHA256SUMS file",
        allow_abbrev=False,
    )
    sources.add_argument("--output-dir", type=Path, required=True)
    check = commands.add_parser(
        "check",
        help="check the texts in a directory for the Day 78 sizes",
        allow_abbrev=False,
    )
    check.add_argument("--source-dir", type=Path, required=True)
    simulate = commands.add_parser(
        "simulate",
        help="predict the bits of both kernels for each input of a Day 78 config",
        allow_abbrev=False,
    )
    simulate.add_argument("--config", type=Path, required=True)
    return parser


def write_sources(directory: Path) -> list[str]:
    """Write each pinned text and the ``SHA256SUMS`` file; return the lines."""

    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise GpuReduceCudaError("output directory must be a directory, not a symlink")
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
        raise GpuReduceCudaError(f"{path.name} must be ASCII text") from error


def check_sources(directory: Path) -> tuple[list[str], list[str]]:
    """Check each pinned file in ``directory``; return its lines and failures.

    A file that is missing or unreadable raises, because no check ran on it.
    """

    kernels = {kernel.file_name: kernel for kernel in PINNED_KERNELS}
    lines, failures = [], []
    for name, pinned in source_files():
        text = _read_text(directory / name)
        results = [
            check_reduce_source(text, kernel=kernels[name], n=n) for n in CHECK_SIZES
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
            f"checks={passed}/{total} sizes={len(CHECK_SIZES)} compiled=false"
        )
    return lines, failures


def _flag(value: bool) -> str:
    return "true" if value else "false"


def prediction_line(prediction: Prediction) -> str:
    """Return the printed line of one predicted result."""

    record = prediction.to_mapping()
    return (
        f"cuda_prediction kernel={record['kernel']} input={record['input']} "
        f"pattern={record['pattern']} n={record['n']} output={record['output']!r} "
        f"bits={record['output_bits']} depth={record['depth']} "
        f"bound_ratio={record['bound_ratio']:.3g} "
        f"exact={_flag(record['exact_result'])} "
        f"race_free={_flag(record['race_free'])} compiled=false"
    )


def simulate_config(path: Path) -> tuple[list[str], list[str]]:
    """Predict both kernels on every input of a config; return lines and failures."""

    config = load_config(path)
    predictions = predict(config)
    lines = [prediction_line(prediction) for prediction in predictions]
    failures = [
        f"{prediction.kernel} on {prediction.input} breaks its bound, counts or log"
        for prediction in predictions
        if not prediction.passed
    ]
    inputs = len({prediction.input for prediction in predictions})
    kernels = len({prediction.kernel for prediction in predictions})
    within = sum(prediction.check.within for prediction in predictions)
    race_free = sum(prediction.to_mapping()["race_free"] for prediction in predictions)
    lines.append(
        f"gpu_reduce_cuda_predictions={predictions_sha256(predictions)} "
        f"config={config.fingerprint()[:12]} kernels={kernels} inputs={inputs} "
        f"within_bounds={within}/{len(predictions)} "
        f"race_free={race_free}/{len(predictions)} compiled=false"
    )
    return lines, failures


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its explicit exit code."""

    arguments = build_parser().parse_args(argv)
    failures: list[str] = []
    try:
        if arguments.command == "sources":
            lines = write_sources(arguments.output_dir)
        elif arguments.command == "check":
            lines, failures = check_sources(arguments.source_dir)
        else:
            lines, failures = simulate_config(arguments.config)
    except (OSError, TypeError, ValueError) as error:
        print(f"gpu-reduce-cuda {arguments.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    for line in lines:
        print(line)
    for failure in failures:
        print(
            f"gpu-reduce-cuda {arguments.command} rejected: {failure}", file=sys.stderr
        )
    return EXIT_REJECTED if failures else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
