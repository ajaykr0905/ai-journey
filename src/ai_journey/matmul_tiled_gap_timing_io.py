"""Load and publish only verified, canonical one-call timing reports.

Reads reuse the bounded reader of ``memory_access_report_io``: regular
non-symlink files only, a byte bound, and strict UTF-8 JSON with unique keys
and an object root. A loaded report must pass the independent verifier and
be byte-for-byte canonical JSON followed by one newline. Writes verify first
and then publish through ``memory_access_report_io.write_report``, which
refuses an output that aliases a protected input such as the config.

Unreadable or malformed files raise ``MatmulTiledGapError``; a report that
was read but fails verification raises ``MatmulTiledGapVerificationError``,
so callers can tell bad input from a rejected report. Nothing here imports
torch.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .matmul_tiled_config import MatmulTiledConfig
from .matmul_tiled_gap_timing import (
    MatmulTiledGapError,
    MatmulTiledGapVerificationError,
)
from .matmul_tiled_gap_timing_verify import verify_timing_report
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import (
    MAX_REPORT_BYTES,
    parse_json_object,
    read_bounded_bytes,
    write_report,
)


def load_timing_report(
    path: Path, *, config: MatmulTiledConfig | None = None
) -> tuple[dict[str, object], dict[str, object]]:
    """Read, verify and canonical-check one timing report; return its summary too.

    Verification runs before the canonical-bytes check, so a tampered report
    is named as such even when it was also reformatted.
    """

    try:
        payload = read_bounded_bytes(path, max_bytes=MAX_REPORT_BYTES, label="report")
        report = parse_json_object(payload, label="report")
    except ValueError as error:
        raise MatmulTiledGapError(str(error)) from error
    summary = verify_timing_report(report, config=config)
    if payload != canonical_json_bytes(report) + b"\n":
        raise MatmulTiledGapVerificationError("report file is not canonical JSON")
    return report, summary


def save_timing_report(
    path: Path,
    report: dict[str, object],
    *,
    config: MatmulTiledConfig | None = None,
    protected: Iterable[Path] = (),
) -> dict[str, object]:
    """Verify a sealed timing report, then publish it atomically; return its summary."""

    summary = verify_timing_report(report, config=config)
    try:
        write_report(path, report, protected=tuple(protected))
    except ValueError as error:
        raise MatmulTiledGapError(str(error)) from error
    return summary
