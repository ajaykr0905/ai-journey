"""Bounded atomic persistence for verified analytical inference reports."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from .inference_seal import canonical_inference_bytes
from .inference_verify import verify_inference_report

MAX_INFERENCE_REPORT_BYTES = 1024 * 1024


def _target(path: str | os.PathLike[str]) -> Path:
    result = Path(path)
    if result.name in ("", ".", ".."):
        raise ValueError("report path must name a file")
    result.parent.mkdir(parents=True, exist_ok=True)
    if result.exists() and (result.is_symlink() or not result.is_file()):
        raise ValueError("report path must be a regular file")
    return result


def write_inference_report(path, report: dict[str, object]) -> None:
    """Verify and atomically replace one bounded JSON report."""

    verify_inference_report(report)
    payload = canonical_inference_bytes(report) + b"\n"
    if len(payload) > MAX_INFERENCE_REPORT_BYTES:
        raise ValueError("inference report exceeds the byte limit")
    target = _target(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise


def read_inference_report(path) -> dict[str, object]:
    """Read one regular bounded file and verify it before returning."""

    target = Path(path)
    metadata = target.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("report path must be a regular file")
    if metadata.st_size > MAX_INFERENCE_REPORT_BYTES:
        raise ValueError("inference report exceeds the byte limit")
    payload = target.read_bytes()
    try:
        report = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("inference report is not valid UTF-8 JSON") from error
    verify_inference_report(report)
    return report
