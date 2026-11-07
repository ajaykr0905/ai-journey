"""Atomic storage for Project 1 lifecycle event chains."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .project1_contract import Project1RunError
from .project1_journal import Project1Event, verify_event_chain


def write_event_chain(path: Path, records: list[Mapping[str, Any]]) -> None:
    """Verify and atomically replace one bounded lifecycle journal."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    verify_event_chain(records)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise Project1RunError("journal parent must be a real directory")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise Project1RunError("journal path must be a regular non-symlink file")
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(encoded) > 1024 * 1024:
        raise Project1RunError("journal exceeds the one MiB storage bound")
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


def load_event_chain(
    path: Path, *, max_bytes: int = 1024 * 1024
) -> tuple[Project1Event, ...]:
    """Load a bounded regular journal and verify every chained record."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if max_bytes <= 0:
        raise Project1RunError("max_bytes must be positive")
    if path.is_symlink() or not path.is_file():
        raise Project1RunError("journal must be a regular non-symlink file")
    if path.stat().st_size > max_bytes:
        raise Project1RunError("journal exceeds max_bytes")
    try:
        records = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise Project1RunError("journal is not valid JSON") from exc
    return verify_event_chain(records)
