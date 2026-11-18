"""Load and publish only verified, canonical profiling reports.

Reads reuse the bounded reader of ``kernel_launch_io``: regular non-symlink
files only, a byte bound, and strict UTF-8 JSON with unique keys and finite
numbers. A loaded report must pass the verifier of its kind and be byte for
byte canonical JSON followed by one newline. Writes verify first, then
publish atomically and refuse a path that aliases a protected input, such
as the config or a snapshot.

Unreadable or malformed files raise ``SimProfileError``. A report that was
read but fails its checks raises ``SimProfileVerificationError``, so a
caller can tell bad input from a rejected report.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from .kernel_launch_io import (
    canonical_json_bytes,
    loads_strict,
    read_bounded,
    write_atomic,
)
from .sim_profile_config import (
    SimProfileConfig,
    SimProfileError,
    SimProfileVerificationError,
)
from .sim_profile_rounds import ROUNDS_KIND
from .sim_profile_rounds_verify import verify_rounds
from .sim_profile_snapshot import SNAPSHOT_KIND
from .sim_profile_verify import verify_snapshot

# A rounds report embeds one snapshot of about 60 KB per stage.
MAX_REPORT_BYTES = 4 * 1024 * 1024


def verify_report(
    report: object, *, config: SimProfileConfig | None = None
) -> dict[str, object]:
    """Check a snapshot or a rounds report with the verifier of its kind."""

    kind = report.get("kind") if type(report) is dict else None
    if kind == SNAPSHOT_KIND:
        return verify_snapshot(report, config=config)
    if kind == ROUNDS_KIND:
        return verify_rounds(report, config=config)
    raise SimProfileVerificationError(
        f"report kind must be {SNAPSHOT_KIND!r} or {ROUNDS_KIND!r}"
    )


def read_report(path: Path, *, label: str = "report") -> dict[str, object]:
    """Read one bounded strict JSON object; the checks come afterwards."""

    try:
        payload = read_bounded(path, limit=MAX_REPORT_BYTES, label=label)
        report = loads_strict(payload, label=label)
    except SimProfileError:
        raise
    except ValueError as error:
        raise SimProfileError(str(error)) from error
    if type(report) is not dict:
        raise SimProfileError(f"{label} root must be a JSON object")
    if payload != canonical_json_bytes(report) + b"\n":
        raise SimProfileVerificationError(f"{label} file is not canonical JSON")
    return report


def load_report(
    path: Path, *, config: SimProfileConfig | None = None, label: str = "report"
) -> tuple[dict[str, object], dict[str, object]]:
    """Read, canonical-check and verify one report; return it with its summary."""

    report = read_report(path, label=label)
    return report, verify_report(report, config=config)


def save_report(
    path: Path, report: Mapping[str, object], *, protected: Iterable[Path] = ()
) -> dict[str, object]:
    """Verify a sealed report, then publish it atomically; return its summary."""

    summary = verify_report(dict(report))
    payload = canonical_json_bytes(report) + b"\n"
    if len(payload) > MAX_REPORT_BYTES:
        raise SimProfileError(f"report exceeds {MAX_REPORT_BYTES} bytes")
    try:
        write_atomic(path, payload, protected=tuple(protected))
    except SimProfileError:
        raise
    except ValueError as error:
        raise SimProfileError(str(error)) from error
    return summary
