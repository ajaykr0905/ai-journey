"""Bounded, atomic persistence for GPT-2 interoperability evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .gpt2_evidence import (
    GPT2Evidence,
    GPT2EvidenceError,
    evidence_payload,
    validate_evidence_payload,
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GPT2EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def write_evidence(path: str | Path, evidence: GPT2Evidence) -> None:
    """Publish canonical evidence atomically in an existing safe directory."""

    if not isinstance(evidence, GPT2Evidence):
        raise TypeError("evidence must be GPT2Evidence")
    destination = Path(path)
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise GPT2EvidenceError(
            "evidence parent must be an existing non-symlink directory"
        )
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise GPT2EvidenceError(
            "evidence destination must not be a symlink or directory"
        )
    encoded = (
        json.dumps(evidence_payload(evidence), indent=2, sort_keys=True) + "\n"
    ).encode()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_evidence(path: str | Path, *, max_bytes: int = 65_536) -> GPT2Evidence:
    """Load bounded strict JSON and verify its evidence digest."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if max_bytes <= 0:
        raise GPT2EvidenceError("max_bytes must be positive")
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise GPT2EvidenceError("evidence must be a regular non-symlink file")
    raw = source.read_bytes()
    if len(raw) > max_bytes:
        raise GPT2EvidenceError("evidence exceeds the byte limit")
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GPT2EvidenceError("evidence must be strict UTF-8 JSON") from exc
    return validate_evidence_payload(payload)
