"""Run and independently verify local worker recovery experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .recovery_config import RecoveryExperimentError
from .recovery_experiment import run_recovery_experiment
from .recovery_report import build_recovery_report
from .recovery_report_io import load_recovery_report, write_recovery_report
from .recovery_verify import verify_recovery_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute a process-kill recovery experiment")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify stored report and trace bytes")
    verify.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the selected recovery command."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "run":
            result = run_recovery_experiment(
                source_path=arguments.source,
                config_path=arguments.config,
                run_dir=arguments.run_dir,
            )
            payload = build_recovery_report(result)
            write_recovery_report(arguments.run_dir / "report.json", payload)
        else:
            payload = load_recovery_report(arguments.run_dir / "report.json")
            verify_recovery_report(payload, trace_path=arguments.run_dir / "trace.json")
    except (OSError, TypeError, ValueError, RecoveryExperimentError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
