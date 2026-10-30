"""Bounded durable storage for distributed capacity evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .distributed_capacity_report import (
    DistributedCapacityReportError,
    verify_distributed_capacity_report,
)

MAX_DISTRIBUTED_CAPACITY_BYTES = 1024 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DistributedCapacityReportError("duplicate capacity report field")
        result[key] = value
    return result


def save_distributed_capacity_report(path: Path, report: dict[str, object]) -> None:
    """Verify and atomically publish a bounded UTF-8 JSON report."""

    if not isinstance(path, Path):
        raise TypeError("capacity report path must be pathlib.Path")
    verify_distributed_capacity_report(report)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_DISTRIBUTED_CAPACITY_BYTES:
        raise DistributedCapacityReportError("capacity report exceeds 1 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
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


def load_distributed_capacity_report(path: Path) -> dict[str, object]:
    """Load bounded evidence while rejecting duplicate fields and tampering."""

    if not isinstance(path, Path):
        raise TypeError("capacity report path must be pathlib.Path")
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_DISTRIBUTED_CAPACITY_BYTES + 1)
        if len(payload) > MAX_DISTRIBUTED_CAPACITY_BYTES:
            raise DistributedCapacityReportError("capacity report exceeds 1 MiB")
        report = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except DistributedCapacityReportError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise DistributedCapacityReportError(
            "unable to read capacity report"
        ) from error
    return verify_distributed_capacity_report(report)
