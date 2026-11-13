"""Atomic live progress observations for recovery workers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .recovery_config import RecoveryExperimentError

_RUN_ID = re.compile(r"[a-z0-9-]+-[0-9a-f]{16}")
_ATTEMPT_ID = re.compile(r"attempt-[1-9][0-9]*-[0-9a-f]{16}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_STATES = {"running", "fault_ready", "completed"}


@dataclass(frozen=True)
class WorkerProgress:
    """Latest process-local progress observed by the controller."""

    run_id: str
    attempt_id: str
    completed_items: int
    accumulator_sha256: str
    monotonic_ns: int
    state: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or _RUN_ID.fullmatch(self.run_id) is None:
            raise RecoveryExperimentError("run_id is invalid")
        if (
            not isinstance(self.attempt_id, str)
            or _ATTEMPT_ID.fullmatch(self.attempt_id) is None
        ):
            raise RecoveryExperimentError("attempt_id is invalid")
        for name in ("completed_items", "monotonic_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise RecoveryExperimentError(f"{name} must be non-negative")
        if (
            not isinstance(self.accumulator_sha256, str)
            or _DIGEST.fullmatch(self.accumulator_sha256) is None
        ):
            raise RecoveryExperimentError("accumulator_sha256 must be a SHA-256 digest")
        if self.state not in _STATES:
            raise RecoveryExperimentError("worker progress state is unsupported")


def write_worker_progress(path: Path, progress: WorkerProgress) -> None:
    """Atomically publish the latest bounded observation."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(progress, WorkerProgress):
        raise TypeError("progress must be WorkerProgress")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise RecoveryExperimentError("progress parent must be a real directory")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RecoveryExperimentError(
            "progress path must be a regular non-symlink file"
        )
    encoded = json.dumps(
        {"schema_version": 1, **asdict(progress)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
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


def load_worker_progress(path: Path, *, max_bytes: int = 16 * 1024) -> WorkerProgress:
    """Load a bounded current observation without accepting unknown fields."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise RecoveryExperimentError("progress must be a bounded regular file")
    try:
        payload = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise RecoveryExperimentError("progress is not valid JSON") from exc
    fields = {"schema_version", *WorkerProgress.__dataclass_fields__}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise RecoveryExperimentError("progress fields do not match schema")
    if payload.pop("schema_version") != 1:
        raise RecoveryExperimentError("progress schema version is unsupported")
    return WorkerProgress(**payload)
