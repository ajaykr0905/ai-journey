"""Run and independently verify loopback serving failure-recovery experiments.

Exit codes: 0 when the evidence verifies and the run recovered (every request
succeeded and the SLO run was observed), 1 when ``verify`` rejects stored
evidence, 2 for invalid inputs or a run that could not complete, and 3 when
the evidence verifies but the run did not recover within its declared bounds.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .serving_recovery_experiment import run_serving_recovery_experiment
from .serving_recovery_report import (
    REPORT_FILE,
    build_serving_report,
    write_serving_report,
)
from .serving_recovery_verify import verify_serving_recovery_run

EXIT_RECOVERED = 0
EXIT_VERIFICATION_FAILED = 1
EXIT_INVALID = 2
EXIT_NOT_RECOVERED = 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-journey-serving-recovery",
        description="Kill a supervised loopback inference server under load, "
        "restart it, and verify measured recovery evidence.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="run one kill-and-recover experiment into a new directory"
    )
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="recompute a stored run from its raw evidence"
    )
    verify.add_argument("--run-dir", type=Path, required=True)
    return parser


def _receipt(receipt: dict) -> int:
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return EXIT_RECOVERED if receipt["recovered"] else EXIT_NOT_RECOVERED


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the selected command and return an explicit exit code."""

    arguments = _parser().parse_args(argv)
    if arguments.command == "verify":
        try:
            receipt = verify_serving_recovery_run(arguments.run_dir)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            print(f"verification failed: {exc}", file=sys.stderr)
            return EXIT_VERIFICATION_FAILED
        return _receipt(receipt)
    try:
        result = run_serving_recovery_experiment(
            source_path=arguments.source,
            config_path=arguments.config,
            run_dir=arguments.run_dir,
        )
        write_serving_report(
            arguments.run_dir / REPORT_FILE,
            build_serving_report(result, run_dir=arguments.run_dir),
            protected=(arguments.source, arguments.config),
        )
        receipt = verify_serving_recovery_run(arguments.run_dir)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    return _receipt(receipt)


if __name__ == "__main__":
    raise SystemExit(main())
