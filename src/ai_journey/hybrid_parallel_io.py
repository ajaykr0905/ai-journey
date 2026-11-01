"""Bounded atomic IO for canonical hybrid-parallel Sealed Reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .hybrid_parallel_report import (
    HybridParallelReportError,
    canonical_hybrid_parallel_report_bytes,
    verify_hybrid_parallel_report,
)


MAX_HYBRID_PARALLEL_REPORT_BYTES = 1024 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HybridParallelReportError("duplicate hybrid parallel report field")
        result[key] = value
    return result


def save_hybrid_parallel_report(path: Path, report: dict[str, object]) -> None:
    """Verify and atomically publish one bounded canonical report."""

    if not isinstance(path, Path):
        raise TypeError("hybrid parallel report path must be pathlib.Path")
    verify_hybrid_parallel_report(report)
    payload = canonical_hybrid_parallel_report_bytes(report)
    if len(payload) > MAX_HYBRID_PARALLEL_REPORT_BYTES:
        raise HybridParallelReportError("hybrid parallel report exceeds 1 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
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


def load_hybrid_parallel_report(path: Path) -> dict[str, object]:
    """Load strict canonical JSON before replaying every derived field."""

    if not isinstance(path, Path):
        raise TypeError("hybrid parallel report path must be pathlib.Path")
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_HYBRID_PARALLEL_REPORT_BYTES + 1)
        if len(payload) > MAX_HYBRID_PARALLEL_REPORT_BYTES:
            raise HybridParallelReportError("hybrid parallel report exceeds 1 MiB")
        report = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(report, dict):
            raise HybridParallelReportError(
                "hybrid parallel report must be a JSON object"
            )
        if payload != canonical_hybrid_parallel_report_bytes(report):
            raise HybridParallelReportError(
                "hybrid parallel report is not canonical JSON"
            )
    except HybridParallelReportError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise HybridParallelReportError(
            "unable to read hybrid parallel report"
        ) from error
    return verify_hybrid_parallel_report(report)
