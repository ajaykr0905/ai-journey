"""Canonical encoding and integrity seals for hybrid-parallel reports."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json

from .hybrid_parallel_schema import HybridParallelReportError


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hybrid_parallel_report_bytes(report: Mapping[str, object]) -> bytes:
    """Return the report's unique UTF-8 JSON representation."""

    if not isinstance(report, Mapping):
        raise TypeError("hybrid parallel report must be a mapping")
    try:
        return _canonical_json_bytes(report)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise HybridParallelReportError(
            "hybrid parallel report cannot be canonically encoded"
        ) from error


def seal_hybrid_parallel_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Attach a recomputable SHA-256 integrity seal without mutating the input."""

    if not isinstance(report, Mapping):
        raise TypeError("hybrid parallel report must be a mapping")
    unsigned = {key: value for key, value in report.items() if key != "integrity"}
    try:
        digest = sha256(_canonical_json_bytes(unsigned)).hexdigest()
        sealed = json.loads(_canonical_json_bytes(unsigned))
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise HybridParallelReportError(
            "hybrid parallel report cannot be canonically encoded"
        ) from error
    sealed["integrity"] = {"algorithm": "sha256", "digest": digest}
    return sealed
