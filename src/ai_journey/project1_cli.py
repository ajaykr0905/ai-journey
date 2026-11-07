"""Operate and verify bounded Project 1 training runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .project1_contract import Project1RunError
from .project1_journal_io import load_event_chain
from .project1_metrics import summarize_step_events
from .project1_operator import (
    execute_fresh_run,
    finalize_run_bundle,
    initialize_run,
    verify_run_bundle,
)
from .project1_state import reduce_run_state


def status_report(run_dir: Path) -> dict[str, Any]:
    """Return durable lifecycle and measured-step state for one run."""

    events = load_event_chain(run_dir / "journal.json")
    payload: dict[str, Any] = {"state": asdict(reduce_run_state(events))}
    steps = [event for event in events if event.kind == "step"]
    payload["metrics"] = asdict(summarize_step_events(steps)) if steps else None
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="admit and initialize a run")
    initialize.add_argument("--source", type=Path, required=True)
    initialize.add_argument("--spec", type=Path, required=True)
    initialize.add_argument("--contract", type=Path, required=True)
    initialize.add_argument("--run-dir", type=Path, required=True)
    initialize.add_argument("--recorded-at-utc")
    execute = commands.add_parser("run", help="execute and finalize a fresh run")
    execute.add_argument("--source", type=Path, required=True)
    execute.add_argument("--spec", type=Path, required=True)
    execute.add_argument("--run-dir", type=Path, required=True)
    execute.add_argument("--recorded-at-utc")
    status = commands.add_parser("status", help="inspect journaled run state")
    status.add_argument("--run-dir", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify a finalized run bundle")
    verify.add_argument("--run-dir", type=Path, required=True)
    return parser


def _timestamp(value: str | None) -> str:
    return value or datetime.now(UTC).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Project 1 operator command line interface."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "init":
            payload = initialize_run(
                source_path=arguments.source,
                spec_path=arguments.spec,
                contract_path=arguments.contract,
                run_dir=arguments.run_dir,
                recorded_at_utc=_timestamp(arguments.recorded_at_utc),
            )
        elif arguments.command == "run":
            execute_fresh_run(
                source_path=arguments.source,
                spec_path=arguments.spec,
                run_dir=arguments.run_dir,
                recorded_at_utc=_timestamp(arguments.recorded_at_utc),
            )
            payload = finalize_run_bundle(arguments.run_dir)
        elif arguments.command == "status":
            payload = status_report(arguments.run_dir)
        else:
            payload = verify_run_bundle(arguments.run_dir)
    except (OSError, TypeError, ValueError, Project1RunError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
