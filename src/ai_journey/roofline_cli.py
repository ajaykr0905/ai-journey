"""Command-line entry point for deterministic decoder roofline reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .resource_accounting import DecoderSpec, ResourceAccountingError
from .roofline_report import (
    RooflinePlan,
    build_roofline_report,
    seal_roofline_report,
)
from .roofline_report_io import (
    MAX_ROOFLINE_REPORT_BYTES,
    load_roofline_report,
    save_roofline_report,
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResourceAccountingError("duplicate roofline plan field")
        result[key] = value
    return result


def load_roofline_plan(path: Path) -> RooflinePlan:
    """Load a bounded JSON plan into validated model and roofline contracts."""

    if not isinstance(path, Path):
        raise TypeError("roofline plan path must be pathlib.Path")
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_ROOFLINE_REPORT_BYTES + 1)
        if len(payload) > MAX_ROOFLINE_REPORT_BYTES:
            raise ResourceAccountingError("roofline plan exceeds 1 MiB")
        raw = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(raw, dict):
            raise TypeError("roofline plan must be an object")
        model = raw.pop("model")
        if not isinstance(model, dict):
            raise TypeError("model must be an object")
        plan = RooflinePlan(model=DecoderSpec(**model), **raw)
        build_roofline_report(plan)
        return plan
    except ResourceAccountingError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as error:
        raise ResourceAccountingError("unable to load roofline plan") from error


def _paths_alias(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve() or (
        first.exists() and second.exists() and first.samefile(second)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and verify analytical decoder roofline evidence; no GPU is run."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a sealed analytical report")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="recompute a sealed report")
    verify.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        report = load_roofline_report(args.report)
        print(
            "verified_roofline="
            f"{report['integrity']['digest']} "
            f"operations={report['summary']['operation_count']} measured=false"
        )
        return 0
    if _paths_alias(args.config, args.output):
        raise ResourceAccountingError(
            "roofline output must not alias its protected config"
        )
    plan = load_roofline_plan(args.config)
    report = seal_roofline_report(build_roofline_report(plan))
    save_roofline_report(args.output, report)
    print(
        "roofline_report="
        f"{report['integrity']['digest']} "
        f"operations={report['summary']['operation_count']} measured=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
