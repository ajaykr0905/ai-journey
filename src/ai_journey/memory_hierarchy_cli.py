"""Run and verify the CPU memory-hierarchy lab; no GPU or CUDA kernel is used.

``run`` loads a config, measures the STREAM working-set sweep, the
access-pattern contrast and the matmul compute ceiling on this CPU host
within the config's runtime budget, derives the roofline and Little's-law
view, verifies the sealed report, and publishes it atomically without ever
writing over the config. A clock injected through ``main`` yields a report
printed and verified as synthetic timing. ``verify`` recomputes a report
independently, optionally against the config it must have been measured
from.

Exit codes: 0 success; 1 a report was read but failed verification; 2 usage
errors, unreadable or invalid inputs, refused outputs, and runs that failed
a kernel check or exceeded their budget.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .memory_access_report_io import paths_alias
from .memory_hierarchy_config import load_config
from .memory_hierarchy_io import load_report, save_report
from .memory_hierarchy_report import measure_report
from .memory_hierarchy_stream import MemoryHierarchyError
from .memory_hierarchy_verify import MemoryHierarchyVerificationError

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2


def build_parser() -> argparse.ArgumentParser:
    """Define the ``run`` and ``verify`` commands with no abbreviations."""

    parser = argparse.ArgumentParser(
        prog="ai-journey-memory-hierarchy",
        description=(
            "Measure STREAM bandwidth across working-set sizes, access-pattern "
            "costs and a NumPy roofline on this CPU host; no GPU or CUDA kernel "
            "is run."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="measure this CPU host and seal a report", allow_abbrev=False
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="recompute a sealed report independently", allow_abbrev=False
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    return parser


def _refuse_unusable_output(output: Path, config: Path) -> None:
    """Refuse an output path the atomic writer would refuse, before measuring.

    A run costs seconds and hundreds of MiB, so an output that aliases the
    config or is a symlink, directory or other non-regular file fails first,
    and so does one whose nearest existing ancestor is not a writable
    directory, where the writer could neither create the missing parents nor
    place its temporary file.
    """

    if paths_alias(output, config):
        raise MemoryHierarchyError("report output must not alias the config")
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise MemoryHierarchyError("report output must be a regular non-symlink file")
    ancestor = output.parent
    # A dangling or looping link stops the walk too, then fails is_dir.
    while not (ancestor.exists() or ancestor.is_symlink()):
        if ancestor.parent == ancestor:
            break
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK | os.X_OK):
        raise MemoryHierarchyError("report output must be inside a writable directory")


def _label(report: dict[str, object]) -> str:
    return "cpu_host_measurement" if report["measured"] else "synthetic_timing"


def _print_run(report: dict[str, object]) -> None:
    label = _label(report)
    stream = report["stream"]
    for name, curve in stream["curves"].items():
        summary = stream["curve_summaries"][name]
        print(
            f"{label} stream kernel={name} points={len(curve)} "
            f"peak_gb_per_second={summary['peak_median_gb_per_second']:.3f} "
            f"peak_working_set_bytes={summary['peak_working_set_bytes']} "
            f"largest_gb_per_second={summary['largest_median_gb_per_second']:.3f} "
            f"largest_working_set_bytes={summary['largest_working_set_bytes']}"
        )
    for item in report["access"]["measurements"]:
        print(
            f"{label} access pattern={item['name']} "
            f"median_ns_per_element={item['median_ns_per_element']:.3f} "
            f"median_gb_per_second={item['median_gb_per_second']:.3f} "
            f"iqr_ns={item['timing']['iqr_ns']}"
        )
    matmul = report["matmul"]
    print(
        f"{label} matmul m={matmul['m']} n={matmul['n']} k={matmul['k']} "
        f"dtype={matmul['dtype']} "
        f"median_gflops_per_second={matmul['median_gflops_per_second']:.3f}"
    )
    roofline = report["roofline"]
    bandwidth = roofline["bandwidth_ceilings"][0]["bytes_per_second"]
    print(
        "derived roofline "
        f"ridge_point_flops_per_byte={roofline['ridge_point_flops_per_byte']:.3f} "
        "compute_ceiling_gflops_per_second="
        f"{roofline['compute_ceiling']['flops_per_second'] / 1e9:.3f} "
        f"bandwidth_ceiling_gb_per_second={bandwidth / 1e9:.3f}"
    )
    law = report["littles_law"]
    random = law["access_patterns"][-1]
    print(
        f"derived littles_law stated_latency_ns={law['stated_latency_ns']} "
        "triad_bytes_in_flight="
        f"{law['bandwidth']['derived_bytes_in_flight']:.1f} "
        f"random_elements_in_flight={random['derived_elements_in_flight']:.2f} "
        "latency_measured=false"
    )


def _summary_line(prefix: str, summary: dict[str, object]) -> str:
    counts = (
        f"stream_measurements={summary['stream_measurements']} "
        f"access_patterns={summary['access_patterns']} scope={summary['scope']}"
    )
    if summary["measured"]:
        return (
            f"{prefix}_measurement={summary['report_sha256']} {counts} "
            "gpu_executed=false"
        )
    return (
        f"{prefix}_synthetic_timing={summary['report_sha256']} {counts} "
        "measured=false gpu_executed=false"
    )


def main(
    argv: Sequence[str] | None = None, *, clock: Callable[[], int] | None = None
) -> int:
    """Run one command and return its exit code; ``clock`` is for tests.

    A replaced clock's durations are reported as synthetic, never measured.
    """

    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            config = load_config(args.config) if args.config is not None else None
            _, summary = load_report(args.report, config=config)
            print(_summary_line("verified_memory_hierarchy", summary))
            return EXIT_OK
        _refuse_unusable_output(args.output, args.config)
        config = load_config(args.config)
        report = measure_report(config, clock=clock or time.perf_counter_ns)
        summary = save_report(
            args.output, report, config=config, protected=(args.config,)
        )
    except MemoryHierarchyVerificationError as error:
        print(f"memory-hierarchy {args.command} rejected: {error}", file=sys.stderr)
        return EXIT_REJECTED
    except (OSError, TypeError, ValueError) as error:
        print(f"memory-hierarchy {args.command} failed: {error}", file=sys.stderr)
        return EXIT_INVALID
    _print_run(report)
    print(_summary_line("memory_hierarchy", summary))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
