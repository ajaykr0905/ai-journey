"""Command-line entry point for deterministic transformer resource plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .resource_accounting import (
    DecoderSpec,
    ResourceAccountingError,
    TrainingWorkload,
)
from .resource_report import ResourcePlan, build_resource_report, seal_resource_report
from .resource_report_io import (
    MAX_RESOURCE_REPORT_BYTES,
    load_resource_report,
    save_resource_report,
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResourceAccountingError("duplicate resource plan field")
        result[key] = value
    return result


def load_resource_plan(path: Path) -> ResourcePlan:
    """Load a bounded JSON plan into validated model and workload contracts."""

    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_RESOURCE_REPORT_BYTES + 1)
        if len(payload) > MAX_RESOURCE_REPORT_BYTES:
            raise ResourceAccountingError("resource plan exceeds 1 MiB")
        raw = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
        model = raw.pop("model")
        workload = raw.pop("workload")
        if not isinstance(model, dict) or not isinstance(workload, dict):
            raise TypeError("model and workload must be objects")
        return ResourcePlan(
            model=DecoderSpec(**model),
            workload=TrainingWorkload(**workload),
            **raw,
        )
    except ResourceAccountingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ResourceAccountingError("unable to load resource plan") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build analytical transformer resource evidence; no GPU is run."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_resource_plan(args.config)
    report = seal_resource_report(build_resource_report(plan))
    save_resource_report(args.output, report)
    print(
        "resource_plan="
        f"{report['integrity']['digest']} "
        f"parameters={report['parameters']['count']} "
        f"fits={str(report['memory']['capacity']['fits']).lower()} "
        "measured=false"
    )
    return 0


def build_verify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and recompute analytical transformer resource evidence."
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser


def verify_main(argv: Sequence[str] | None = None) -> int:
    args = build_verify_parser().parse_args(argv)
    report = load_resource_report(args.report)
    print(
        "verified_resource_plan="
        f"{report['integrity']['digest']} "
        f"parameters={report['parameters']['count']} "
        "measured=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
