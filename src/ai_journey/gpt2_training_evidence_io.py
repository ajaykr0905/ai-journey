"""Bounded atomic storage for GPT-2 training evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .gpt2_training import GPT2TrainingError
from .gpt2_training_evidence import verify_training_evidence


def write_training_evidence(path: Path, payload: Mapping[str, Any]) -> None:
    """Validate, atomically replace, and durably publish one JSON report."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    verified = verify_training_evidence(payload)
    encoded = (
        json.dumps(verified, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_training_evidence(
    path: Path, *, max_bytes: int = 16 * 1024 * 1024
) -> dict[str, Any]:
    """Load and verify a bounded regular UTF-8 JSON report."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if max_bytes <= 0:
        raise GPT2TrainingError("max_bytes must be positive")
    if path.is_symlink() or not path.is_file():
        raise GPT2TrainingError("training evidence must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise GPT2TrainingError("training evidence exceeds its byte budget")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GPT2TrainingError("training evidence must be valid UTF-8 JSON") from exc
    return verify_training_evidence(payload)
