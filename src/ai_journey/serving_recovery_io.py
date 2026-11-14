"""Atomic bounded persistence for serving recovery evidence.

Writers render canonical JSON into a temporary file beside the destination,
flush and fsync it, then hard-link it into an unused path. Unlike ``replace``,
``link`` fails if the destination appeared meanwhile, so neither evidence nor a
declared protected input is ever overwritten. Loaders read bounded regular
files, require canonical strict JSON and validate every record they return.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .serving_recovery_client import AttemptRecord, attempt_from_mapping
from .serving_recovery_config import (
    ServingRecoveryError,
    canonical_json_bytes,
    read_bounded_file,
    require_int,
    strict_json_loads,
)
from .serving_recovery_trace import ServingLifecycleEvent, verify_serving_trace

MAX_ATTEMPT_LOG_BYTES = 8 * 1024 * 1024
MAX_TRACE_BYTES = 256 * 1024


def publish_new_file(
    path: Path, data: bytes, *, max_bytes: int, protected: Sequence[Path] = ()
) -> None:
    """Atomically publish complete bytes at a path that does not exist yet."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if type(data) is not bytes:
        raise TypeError("data must be bytes")
    require_int(max_bytes, "max_bytes", 1, MAX_ATTEMPT_LOG_BYTES)
    if len(data) > max_bytes:
        raise ServingRecoveryError(f"{path.name} exceeds its {max_bytes}-byte bound")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ServingRecoveryError("evidence parent must be a real directory")
    target = parent.resolve() / path.name
    if any(target == Path(item).resolve() for item in protected):
        raise ServingRecoveryError(f"refusing to overwrite protected {path.name}")
    if os.path.lexists(path):
        raise ServingRecoveryError(f"refusing to overwrite {path.name}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ServingRecoveryError(f"refusing to overwrite {path.name}") from exc
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_canonical_json(path: Path, *, max_bytes: int, name: str) -> Any:
    """Load bounded strict JSON whose bytes are exactly its canonical form."""

    data = read_bounded_file(path, max_bytes=max_bytes, name=name)
    payload = strict_json_loads(data, name=name)
    if canonical_json_bytes(payload) != data:
        raise ServingRecoveryError(f"{name} is not canonical JSON")
    return payload


def write_attempt_log(
    path: Path, attempts: Sequence[AttemptRecord], *, protected: Sequence[Path] = ()
) -> None:
    """Persist every classified client attempt in ledger order."""

    if any(not isinstance(attempt, AttemptRecord) for attempt in attempts):
        raise TypeError("attempts must be AttemptRecord values")
    payload = {
        "schema_version": 1,
        "attempts": [attempt.to_mapping() for attempt in attempts],
    }
    publish_new_file(
        path,
        canonical_json_bytes(payload),
        max_bytes=MAX_ATTEMPT_LOG_BYTES,
        protected=protected,
    )


def load_attempt_log(path: Path) -> tuple[AttemptRecord, ...]:
    """Load and validate a persisted attempt log."""

    payload = load_canonical_json(
        path, max_bytes=MAX_ATTEMPT_LOG_BYTES, name="attempt log"
    )
    if type(payload) is not dict or set(payload) != {"schema_version", "attempts"}:
        raise ServingRecoveryError("attempt log fields do not match schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ServingRecoveryError("attempt log schema is unsupported")
    if type(payload["attempts"]) is not list:
        raise ServingRecoveryError("attempt log attempts must be a list")
    return tuple(attempt_from_mapping(item) for item in payload["attempts"])


def write_trace(
    path: Path, records: Sequence[dict[str, Any]], *, protected: Sequence[Path] = ()
) -> None:
    """Verify a chained lifecycle trace, then persist it."""

    verify_serving_trace(records, require_finished=False)
    publish_new_file(
        path,
        canonical_json_bytes(list(records)),
        max_bytes=MAX_TRACE_BYTES,
        protected=protected,
    )


def load_trace(
    path: Path, *, require_finished: bool = True
) -> tuple[list[dict[str, Any]], tuple[ServingLifecycleEvent, ...]]:
    """Load a persisted trace and verify its whole chain and lifecycle."""

    records = load_canonical_json(path, max_bytes=MAX_TRACE_BYTES, name="trace")
    if type(records) is not list:
        raise ServingRecoveryError("trace must be a JSON list")
    return records, verify_serving_trace(records, require_finished=require_finished)
