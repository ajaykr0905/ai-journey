"""Load and publish only verified, canonical checklist bound reports.

Reads reuse the bounded reader of ``memory_access_report_io``: regular
non-symlink files only, a byte bound, and strict UTF-8 JSON with unique keys
and an object root. A loaded report must pass the verifier and be
byte-for-byte canonical JSON followed by one newline. Given the three source
paths, the loader also rebuilds the bound through the sources' own verified
loaders with the rule config, and the report must equal that rebuild. Writes
verify first and then publish through ``memory_access_report_io.write_report``,
which refuses an output that aliases a protected input such as a source.

Unreadable or malformed files raise ``MatmulChecklistBoundError``; a report
that was read but fails verification raises
``MatmulChecklistBoundVerificationError``, so callers can tell bad input
from a rejected report. Nothing here imports torch.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .matmul_bottleneck_config import BottleneckConfig
from .matmul_checklist_bound import bound_from_files
from .matmul_checklist_bound_traffic import (
    MatmulChecklistBoundError,
    MatmulChecklistBoundVerificationError,
)
from .matmul_checklist_bound_verify import verify_bound_report
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import (
    MAX_REPORT_BYTES,
    parse_json_object,
    read_bounded_bytes,
    write_report,
)


@dataclass(frozen=True)
class BoundSources:
    """The paths of the three source reports of one bound."""

    checklist_report: Path
    ceiling_report: Path
    timing_report: Path

    def __post_init__(self) -> None:
        if not all(isinstance(path, Path) for path in self.all()):
            raise TypeError("source report paths must be pathlib.Path")

    def all(self) -> tuple[Path, Path, Path]:
        return self.checklist_report, self.ceiling_report, self.timing_report


def load_bound_report(
    path: Path,
    *,
    config: BottleneckConfig | None = None,
    sources: BoundSources | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Read, verify and canonical-check one bound report; return its summary too.

    The report's own checks run before the rebuild from the sources, so a
    tampered bound report is named as such before any source is read. A
    rebuild needs the rule config, so a caller with sources must give it.
    """

    if sources is not None and config is None:
        raise MatmulChecklistBoundError(
            "a rebuild from the sources needs the rule config"
        )
    try:
        payload = read_bounded_bytes(path, max_bytes=MAX_REPORT_BYTES, label="report")
        report = parse_json_object(payload, label="report")
    except ValueError as error:
        raise MatmulChecklistBoundError(str(error)) from error
    summary = verify_bound_report(report, config=config)
    if sources is not None:
        rebuilt = bound_from_files(*sources.all(), config)
        summary = verify_bound_report(report, config=config, rebuilt=rebuilt)
    if payload != canonical_json_bytes(report) + b"\n":
        raise MatmulChecklistBoundVerificationError("report file is not canonical JSON")
    return report, summary


def save_bound_report(
    path: Path,
    report: dict[str, object],
    *,
    config: BottleneckConfig | None = None,
    protected: Iterable[Path] = (),
) -> dict[str, object]:
    """Verify a sealed bound report, then publish it atomically; return its summary.

    Protected paths keep the bound from overwriting a source or the rule config.
    """

    summary = verify_bound_report(report, config=config)
    try:
        write_report(path, report, protected=tuple(protected))
    except ValueError as error:
        raise MatmulChecklistBoundError(str(error)) from error
    return summary
