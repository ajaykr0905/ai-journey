"""Export and verify local CPU lab telemetry derived from recovery evidence.

``export`` verifies one serving-recovery run directory and one worker-recovery
run directory with their own verifiers, derives Prometheus text exposition
0.0.4 and recording and alerting rules from the declared objectives, seals a
report, publishes everything into a new directory and then verifies what it
wrote with the code ``verify`` uses. ``verify`` recomputes a stored export
from the evidence directories and requires every byte to match; given
``--config``, it also requires the export to use exactly those controls. No
server is scraped and nothing is sent anywhere: both commands work on local
files only, and only ``export`` writes, into its new output directory.

Exit codes: 0 when the export verifies and every declared objective is met,
1 when ``verify`` rejects an export or its evidence, 2 for usage errors,
invalid inputs, refused outputs and evidence that ``export`` rejects, and 3
when the export verifies but a declared objective is breached, so its alert
would fire.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .telemetry_metrics_config import load_telemetry_metrics_config
from .telemetry_metrics_report import build_metrics_export, write_metrics_export
from .telemetry_metrics_verify import verify_metrics_export

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_INVALID = 2
EXIT_OBJECTIVE_BREACHED = 3


def build_parser() -> argparse.ArgumentParser:
    """Define the ``export`` and ``verify`` commands without abbreviations."""

    parser = argparse.ArgumentParser(
        prog="python -m ai_journey.telemetry_metrics_cli",
        description=(
            "Derive Prometheus metrics and SLO rules from verified local CPU "
            "recovery evidence, or verify a stored export; no server is scraped."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "export",
        help="verify evidence and publish metrics, rules and a sealed report",
        allow_abbrev=False,
    )
    export.add_argument("--config", type=Path, required=True)
    export.add_argument("--serving-run-dir", type=Path, required=True)
    export.add_argument("--worker-run-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser(
        "verify",
        help="recompute a stored export from its evidence",
        allow_abbrev=False,
    )
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument("--serving-run-dir", type=Path, required=True)
    verify.add_argument("--worker-run-dir", type=Path, required=True)
    verify.add_argument(
        "--config", type=Path, help="require the export to use these controls"
    )
    return parser


def _receipt(receipt: dict[str, Any]) -> int:
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return EXIT_OBJECTIVE_BREACHED if receipt["breached_objectives"] else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the selected command and return an explicit exit code."""

    arguments = build_parser().parse_args(argv)
    evidence = {
        "serving_run_dir": arguments.serving_run_dir,
        "worker_run_dir": arguments.worker_run_dir,
    }
    if arguments.command == "verify":
        try:
            declared = (
                None
                if arguments.config is None
                else load_telemetry_metrics_config(arguments.config)
            )
        except (OSError, TypeError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return EXIT_INVALID
        try:
            receipt = verify_metrics_export(
                arguments.output_dir, **evidence, declared_config=declared
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(f"verification failed: {error}", file=sys.stderr)
            return EXIT_REJECTED
        return _receipt(receipt)
    try:
        config = load_telemetry_metrics_config(arguments.config)
        export = build_metrics_export(config, **evidence)
        write_metrics_export(
            arguments.output_dir, export, evidence_dirs=tuple(evidence.values())
        )
        receipt = verify_metrics_export(arguments.output_dir, **evidence)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_INVALID
    return _receipt(receipt)


if __name__ == "__main__":
    raise SystemExit(main())
