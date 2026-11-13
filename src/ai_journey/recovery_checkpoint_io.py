"""Atomic persistence for sealed recovery checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .recovery_checkpoint import RecoveryCheckpoint, checkpoint_from_mapping
from .recovery_config import RecoveryExperimentError


def write_recovery_checkpoint(path: Path, checkpoint: RecoveryCheckpoint) -> str:
    """Atomically publish a canonical checkpoint and return its digest."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(checkpoint, RecoveryCheckpoint):
        raise TypeError("checkpoint must be RecoveryCheckpoint")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise RecoveryExperimentError("checkpoint parent must be a real directory")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RecoveryExperimentError(
            "checkpoint path must be a regular non-symlink file"
        )
    payload = checkpoint.to_mapping()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
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
    return payload["checkpoint_sha256"]


def load_recovery_checkpoint(
    path: Path, *, max_bytes: int = 64 * 1024
) -> RecoveryCheckpoint:
    """Load a bounded regular checkpoint and verify its content seal."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if max_bytes <= 0:
        raise RecoveryExperimentError("max_bytes must be positive")
    if path.is_symlink() or not path.is_file():
        raise RecoveryExperimentError("checkpoint must be a regular non-symlink file")
    if path.stat().st_size > max_bytes:
        raise RecoveryExperimentError("checkpoint exceeds max_bytes")
    try:
        payload = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise RecoveryExperimentError("checkpoint is not valid JSON") from exc
    return checkpoint_from_mapping(payload)
