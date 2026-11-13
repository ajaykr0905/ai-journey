"""Atomic persistence for hash-chained recovery traces."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .recovery_config import RecoveryExperimentError
from .recovery_trace import RecoveryEvent, chain_recovery_event
from .recovery_trace_verify import verify_recovery_trace


def write_recovery_trace(path: Path, records: list[dict[str, object]]) -> None:
    """Verify and atomically replace one bounded trace."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    verify_recovery_trace(records)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise RecoveryExperimentError("trace parent must be a real directory")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RecoveryExperimentError("trace path must be a regular non-symlink file")
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(encoded) > 1024 * 1024:
        raise RecoveryExperimentError("trace exceeds the one MiB storage bound")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_recovery_trace(
    path: Path, *, max_bytes: int = 1024 * 1024, require_completed: bool = True
):
    """Load a bounded regular trace and verify its entire chain."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if max_bytes <= 0:
        raise RecoveryExperimentError("max_bytes must be positive")
    if path.is_symlink() or not path.is_file():
        raise RecoveryExperimentError("trace must be a regular non-symlink file")
    if path.stat().st_size > max_bytes:
        raise RecoveryExperimentError("trace exceeds max_bytes")
    try:
        records = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise RecoveryExperimentError("trace is not valid JSON") from exc
    return verify_recovery_trace(records, require_completed=require_completed)


def append_recovery_event(
    path: Path, event: RecoveryEvent
) -> tuple[RecoveryEvent, ...]:
    """Append one event while preserving sequence and hash ancestry."""

    if not isinstance(event, RecoveryEvent):
        raise TypeError("event must be RecoveryEvent")
    if path.exists():
        existing = load_recovery_trace(path, require_completed=False)
        records = json.loads(path.read_bytes())
        if existing[-1].kind == "completed":
            raise RecoveryExperimentError("completed traces cannot be extended")
    else:
        existing, records = (), []
    if event.sequence != len(existing):
        raise RecoveryExperimentError("appended event sequence is not contiguous")
    previous = records[-1]["event_sha256"] if records else None
    records.append(chain_recovery_event(event, previous))
    verify_recovery_trace(records, require_completed=event.kind == "completed")
    _write_trace_bytes(path, records)
    return (*existing, event)


def _write_trace_bytes(path: Path, records: list[dict[str, object]]) -> None:
    """Atomically store an already verified trace or trace prefix."""

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise RecoveryExperimentError("trace parent must be a real directory")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RecoveryExperimentError("trace path must be a regular non-symlink file")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > 1024 * 1024:
        raise RecoveryExperimentError("trace exceeds the one MiB storage bound")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
