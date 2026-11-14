"""Canonical SHA-256 sealing for analytical inference reports."""

from __future__ import annotations

import json
from hashlib import sha256


def canonical_inference_bytes(report: dict[str, object]) -> bytes:
    """Return the single byte encoding that sealing and verification compare.

    Sorted keys and compact separators make it deterministic; NaN or infinity
    raise ValueError and non-dict reports raise TypeError.
    """

    if type(report) is not dict:
        raise TypeError("report must be a dictionary")
    return json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def inference_report_digest(report: dict[str, object]) -> str:
    """Return the SHA-256 of every field except ``report_sha256`` itself."""

    payload = dict(report)
    payload.pop("report_sha256", None)
    return sha256(canonical_inference_bytes(payload)).hexdigest()


def seal_inference_report(report: dict[str, object]) -> dict[str, object]:
    """Return a copy with a digest over every other report field."""

    if "report_sha256" in report:
        raise ValueError("report is already sealed")
    sealed = dict(report)
    sealed["report_sha256"] = inference_report_digest(sealed)
    return sealed
