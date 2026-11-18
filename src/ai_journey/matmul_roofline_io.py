"""Load and publish only verified, canonical gap reports.

Reads reuse the bounded reader of ``memory_access_report_io``: regular
non-symlink files only, a byte bound, and strict UTF-8 JSON with unique keys
and an object root. A loaded report must pass the verifier and be
byte-for-byte canonical JSON followed by one newline. Given both source
paths, the loader also rebuilds the gap through the sources' own verified
loaders, and the report must equal that rebuild. Writes verify first and then
publish through ``memory_access_report_io.write_report``, which refuses an
output that aliases a protected input such as a source report.

Unreadable or malformed files raise ``MatmulRooflineError``; a report that was
read but fails verification raises ``MatmulRooflineVerificationError``, so
callers can tell bad input from a rejected report. Nothing here imports torch.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .matmul_roofline_gap import (
    MatmulRooflineError,
    MatmulRooflineVerificationError,
    gap_from_files,
)
from .matmul_roofline_verify import verify_gap_report
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import (
    MAX_REPORT_BYTES,
    parse_json_object,
    read_bounded_bytes,
    write_report,
)


def load_report(
    path: Path,
    *,
    matmul_report: Path | None = None,
    ceiling_report: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Read, verify and canonical-check one gap report; return it with its summary.

    The report's own checks run before the rebuild from the sources, so a
    tampered gap report is named as such before any source is read.
    """

    if (matmul_report is None) != (ceiling_report is None):
        raise MatmulRooflineError(
            "a gap report binds to both source reports or to neither"
        )
    try:
        payload = read_bounded_bytes(path, max_bytes=MAX_REPORT_BYTES, label="report")
        report = parse_json_object(payload, label="report")
    except ValueError as error:
        raise MatmulRooflineError(str(error)) from error
    summary = verify_gap_report(report)
    if matmul_report is not None:
        rebuilt = gap_from_files(matmul_report, ceiling_report)
        summary = verify_gap_report(report, rebuilt=rebuilt)
    if payload != canonical_json_bytes(report) + b"\n":
        raise MatmulRooflineVerificationError("report file is not canonical JSON")
    return report, summary


def save_report(
    path: Path, report: dict[str, object], *, protected: Iterable[Path] = ()
) -> dict[str, object]:
    """Verify a sealed gap report, then publish it atomically; return its summary.

    Protected paths keep the gap from overwriting either source report.
    """

    summary = verify_gap_report(report)
    try:
        write_report(path, report, protected=tuple(protected))
    except ValueError as error:
        raise MatmulRooflineError(str(error)) from error
    return summary
