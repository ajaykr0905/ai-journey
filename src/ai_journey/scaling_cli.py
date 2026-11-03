"""Command-line interface for buildable and independently verified scaling plans."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .scaling_io import load_scaling_report, save_scaling_report
from .scaling_laws import ScalingLawError
from .scaling_protocol import load_scaling_protocol
from .scaling_report import build_scaling_report
from .scaling_verify import seal_scaling_report


def _paths_alias(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve() or (
        first.exists() and second.exists() and first.samefile(second)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and verify analytical scaling plans; no training or GPU is run."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a sealed analytical report")
    build.add_argument("--protocol", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="recompute a sealed report")
    verify.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        report = load_scaling_report(args.report)
        print(
            "verified_scaling="
            f"{report['integrity']['digest']} "
            f"parameters={report['protocol']['parameter_count']} "
            f"tokens={report['target']['tokens']} measured=false"
        )
        return 0
    if _paths_alias(args.protocol, args.output):
        raise ScalingLawError("scaling output must not alias its protected protocol")
    protocol = load_scaling_protocol(args.protocol)
    report = seal_scaling_report(build_scaling_report(protocol))
    save_scaling_report(args.output, report)
    print(
        "scaling_report="
        f"{report['integrity']['digest']} "
        f"parameters={report['protocol']['parameter_count']} "
        f"tokens={report['target']['tokens']} measured=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
