"""Bounded atomic storage for verified analytical scaling reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .scaling_laws import ScalingLawError
from .scaling_verify import verify_scaling_report

MAX_SCALING_REPORT_BYTES = 1024 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ScalingLawError("duplicate scaling report field")
        result[key] = value
    return result


def save_scaling_report(path: Path, report: dict[str, object]) -> None:
    """Verify, flush, and atomically replace one bounded JSON report."""

    if not isinstance(path, Path):
        raise TypeError("scaling report path must be pathlib.Path")
    verify_scaling_report(report)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_SCALING_REPORT_BYTES:
        raise ScalingLawError("scaling report exceeds 1 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_scaling_report(path: Path) -> dict[str, object]:
    """Load and verify one bounded report, rejecting duplicate JSON fields."""

    if not isinstance(path, Path):
        raise TypeError("scaling report path must be pathlib.Path")
    try:
        if path.is_symlink() or not path.is_file():
            raise ScalingLawError("scaling report must be a regular file")
        with path.open("rb") as stream:
            payload = stream.read(MAX_SCALING_REPORT_BYTES + 1)
        if len(payload) > MAX_SCALING_REPORT_BYTES:
            raise ScalingLawError("scaling report exceeds 1 MiB")
        report = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except ScalingLawError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ScalingLawError("unable to read scaling report") from error
    return verify_scaling_report(report)
