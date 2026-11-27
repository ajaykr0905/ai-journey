"""Load and publish only verified, canonical Day 82 reports.

Reads reuse the bounded reader of ``memory_access_report_io``: regular
non-symlink files only, a byte bound, and strict UTF-8 JSON with unique keys
and an object root. A loaded report must pass the independent verifier and
be byte-for-byte canonical JSON followed by one newline. Writes verify first
and then publish through ``memory_access_report_io.write_report``, which
writes a fsynced temporary file in the target directory, moves it into place
with ``os.replace``, and refuses an output that aliases a protected input
such as the config or the ceiling report of the run.

Unreadable or malformed files raise ``MetalTcError``; a report that was
read but fails verification raises ``MetalTcVerificationError``, so callers
can tell bad input from a rejected report. Nothing here imports torch.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import (
    MAX_REPORT_BYTES,
    parse_json_object,
    read_bounded_bytes,
    write_report,
)
from .metal_flash_ceiling import FlashCeiling
from .metal_tc_ceiling import load_tc_ceiling
from .metal_tc_config import MetalTcConfig
from .metal_tc_source import MetalTcError
from .metal_tc_verify import MetalTcVerificationError, verify_report


def load_report(
    path: Path,
    *,
    config: MetalTcConfig | None = None,
    ceiling: FlashCeiling | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Read, verify and canonical-check one report; return it with its summary.

    Verification runs before the canonical-bytes check, so a tampered report
    is named as such even when it was also reformatted.
    """

    try:
        payload = read_bounded_bytes(path, max_bytes=MAX_REPORT_BYTES, label="report")
        report = parse_json_object(payload, label="report")
    except ValueError as error:
        raise MetalTcError(str(error)) from error
    summary = verify_report(report, config=config, ceiling=ceiling)
    if payload != canonical_json_bytes(report) + b"\n":
        raise MetalTcVerificationError("report file is not canonical JSON")
    return report, summary


def save_report(
    path: Path,
    report: dict[str, object],
    *,
    config: MetalTcConfig | None = None,
    ceiling: FlashCeiling | None = None,
    protected: Iterable[Path] = (),
) -> dict[str, object]:
    """Verify a sealed report, then publish it atomically; return its summary.

    Self-verification means no file this lab writes can fail its own
    verifier, and protected paths keep a run from overwriting its inputs.
    """

    summary = verify_report(report, config=config, ceiling=ceiling)
    try:
        write_report(path, report, protected=tuple(protected))
    except ValueError as error:
        raise MetalTcError(str(error)) from error
    return summary


def read_ceiling(path: Path) -> FlashCeiling:
    """Read a Day 74 gpu-ceilings report through the verified Day 74 loader."""

    return load_tc_ceiling(path)
