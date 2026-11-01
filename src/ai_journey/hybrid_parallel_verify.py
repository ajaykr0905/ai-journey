"""Offline verification for a hybrid-parallel Sealed Report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from .hybrid_parallel_io import load_hybrid_parallel_report
from .hybrid_parallel_report import HybridParallelReportError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = load_hybrid_parallel_report(args.report)
    except (HybridParallelReportError, OSError, TypeError, ValueError) as error:
        print(f"hybrid parallel verification failed: {error}", file=sys.stderr)
        return 2
    print(
        f"verified_hybrid_parallel={report['integrity']['digest']} "
        "recorded=cpu_gloo execution_authenticity=unverified "
        "seal=integrity_and_internal_consistency performance=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
