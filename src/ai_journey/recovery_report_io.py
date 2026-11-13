"""Atomic persistence for canonical recovery reports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .recovery_config import RecoveryExperimentError


def write_recovery_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically write one bounded canonical JSON report."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(report, dict) or not report:
        raise TypeError("report must be a non-empty mapping")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise RecoveryExperimentError("report parent must be a real directory")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RecoveryExperimentError("report path must be a regular non-symlink file")
    encoded = json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(encoded) > 256 * 1024:
        raise RecoveryExperimentError("report exceeds the 256 KiB storage bound")
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


def load_recovery_report(path: Path, *, max_bytes: int = 256 * 1024) -> dict[str, Any]:
    """Load bounded report bytes without following aliases."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if max_bytes <= 0:
        raise RecoveryExperimentError("max_bytes must be positive")
    if path.is_symlink() or not path.is_file():
        raise RecoveryExperimentError("report must be a regular non-symlink file")
    if path.stat().st_size > max_bytes:
        raise RecoveryExperimentError("report exceeds max_bytes")
    try:
        payload = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise RecoveryExperimentError("report is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RecoveryExperimentError("report root must be a mapping")
    return payload
