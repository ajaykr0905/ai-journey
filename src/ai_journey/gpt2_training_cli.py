"""Installed commands for deterministic GPT-2 training and evidence replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .gpt2_training import (
    GPT2TrainingError,
    load_training_spec,
    run_training_experiment,
)
from .gpt2_training_evidence import (
    build_training_evidence,
    verify_training_replay,
)
from .gpt2_training_evidence_io import (
    load_training_evidence,
    write_training_evidence,
)


def load_source_bytes(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
    """Read a bounded regular source file without following symlinks."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if max_bytes <= 0:
        raise GPT2TrainingError("max_bytes must be positive")
    if path.is_symlink() or not path.is_file():
        raise GPT2TrainingError("source must be a regular non-symlink file")
    size = path.stat().st_size
    if size < 2 or size > max_bytes:
        raise GPT2TrainingError("source exceeds its byte limits")
    return path.read_bytes()


def run_and_publish(
    *,
    source_path: Path,
    spec_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    resume: bool = False,
    max_source_bytes: int = 16 * 1024 * 1024,
    max_parameters: int = 5_000_000,
) -> dict[str, Any]:
    """Run the declared experiment and publish verified JSON evidence."""

    source = load_source_bytes(source_path, max_bytes=max_source_bytes)
    spec = load_training_spec(spec_path)
    run = run_training_experiment(
        source,
        spec,
        checkpoint_path=checkpoint_path,
        resume=resume,
        max_parameters=max_parameters,
    )
    evidence = build_training_evidence(run)
    write_training_evidence(report_path, evidence)
    return evidence


def verify_report(path: Path) -> dict[str, Any]:
    """Verify a saved report without repeating model execution."""

    return load_training_evidence(path)


def replay_report(*, source_path: Path, report_path: Path) -> dict[str, Any]:
    """Repeat a saved experiment and require byte-for-byte evidence equality."""

    source = load_source_bytes(source_path)
    report = load_training_evidence(report_path)
    return verify_training_replay(source, report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="run or resume training")
    train.add_argument("--source", type=Path, required=True)
    train.add_argument("--spec", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--report", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--max-source-bytes", type=int, default=16 * 1024 * 1024)
    train.add_argument("--max-parameters", type=int, default=5_000_000)
    verify = commands.add_parser("verify", help="verify saved evidence")
    verify.add_argument("--report", type=Path, required=True)
    replay = commands.add_parser("replay", help="repeat and compare a saved run")
    replay.add_argument("--source", type=Path, required=True)
    replay.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the GPT-2 training command line interface."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "train":
            payload = run_and_publish(
                source_path=arguments.source,
                spec_path=arguments.spec,
                checkpoint_path=arguments.checkpoint,
                report_path=arguments.report,
                resume=arguments.resume,
                max_source_bytes=arguments.max_source_bytes,
                max_parameters=arguments.max_parameters,
            )
        elif arguments.command == "verify":
            payload = verify_report(arguments.report)
        else:
            payload = replay_report(
                source_path=arguments.source,
                report_path=arguments.report,
            )
    except (GPT2TrainingError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
