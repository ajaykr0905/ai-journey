"""Bounded, atomic persistence for sealed serving evidence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .serving_report import (
    SealedServingReport,
    ServingReportError,
    sealed_report_payload,
    verify_serving_document,
)

MAX_REPORT_BYTES = 1_048_576


def write_sealed_report(path: Path, sealed: SealedServingReport) -> Path:
    """Atomically replace a report after writing canonical JSON beside it."""

    if not isinstance(path, Path):
        raise TypeError("path must be Path")
    if not isinstance(sealed, SealedServingReport):
        raise TypeError("sealed must be SealedServingReport")
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(
        sealed_report_payload(sealed),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(document) > MAX_REPORT_BYTES:
        raise ServingReportError("serving report exceeds size limit")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def read_sealed_report(
    path: Path, *, max_bytes: int = MAX_REPORT_BYTES
) -> SealedServingReport:
    """Read and verify a regular, size-bounded sealed report."""

    if not isinstance(path, Path):
        raise TypeError("path must be Path")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ServingReportError("max_bytes must be positive")
    if path.is_symlink() or not path.is_file():
        raise ServingReportError("serving report must be a regular file")
    if path.stat().st_size > max_bytes:
        raise ServingReportError("serving report exceeds size limit")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ServingReportError("serving report is not valid JSON") from error
    return verify_serving_document(payload)
