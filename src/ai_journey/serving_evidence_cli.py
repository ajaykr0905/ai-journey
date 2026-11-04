"""Command-line verification for sealed serving evidence reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .serving_report import ServingReportError
from .serving_report_io import read_sealed_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-journey-serving-evidence",
        description="Verify an immutable serving evidence report.",
    )
    parser.add_argument("report", type=Path, help="path to a sealed JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Verify one report and emit a small machine-readable receipt."""

    args = build_parser().parse_args(argv)
    try:
        sealed = read_sealed_report(args.report)
    except ServingReportError as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 1
    receipt = {
        "engine_label": sealed.report.snapshot.engine_label,
        "execution_kind": sealed.report.boundary.execution_kind,
        "report_sha256": sealed.report_sha256,
        "verified": True,
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
