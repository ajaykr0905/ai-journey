"""Atomic, bounded publication and loading of verified SIMT reports."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import stat
import tempfile

from .simt_device import SimtError
from .simt_json import canonical_json_bytes, loads_strict, read_bounded
from .simt_verify import verify_simt_report

MAX_SIMT_REPORT_BYTES = 1024 * 1024


def _aliases(first: Path, second: Path) -> bool:
    try:
        same_target = first.resolve() == second.resolve()
    except (OSError, RuntimeError) as error:
        # Python 3.11 raises RuntimeError when a path is a symlink loop.
        raise SimtError(
            "SIMT report paths must resolve without a symlink loop"
        ) from error
    if same_target:
        return True
    try:
        return first.exists() and second.exists() and first.samefile(second)
    except OSError:
        return False


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_simt_report(
    path: Path, report: dict[str, object], *, protected: Sequence[Path] = ()
) -> None:
    """Verify, then atomically replace ``path`` with canonical report bytes.

    The write refuses any target that resolves to, or is a hard link of, a
    protected input such as the plan the report was built from.
    """

    if not isinstance(path, Path):
        raise TypeError("SIMT report path must be pathlib.Path")
    if path.name in ("", ".", ".."):
        raise SimtError("SIMT report path must name a file")
    if any(not isinstance(item, Path) for item in protected):
        raise TypeError("protected inputs must be pathlib.Path values")
    verify_simt_report(report)
    payload = canonical_json_bytes(report) + b"\n"
    if len(payload) > MAX_SIMT_REPORT_BYTES:
        raise SimtError(f"SIMT report exceeds {MAX_SIMT_REPORT_BYTES} bytes")
    if any(_aliases(path, item) for item in protected):
        raise SimtError("SIMT report output must not alias a protected input")
    if path.is_symlink() or (path.exists() and not stat.S_ISREG(path.stat().st_mode)):
        raise SimtError("SIMT report output must be a regular non-symlink file")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        stream = os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        Path(temporary).unlink(missing_ok=True)
        raise
    try:
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def load_simt_report(path: Path) -> dict[str, object]:
    """Load canonical, duplicate-free report bytes and verify every field."""

    payload = read_bounded(path, limit=MAX_SIMT_REPORT_BYTES, label="SIMT report")
    report = loads_strict(payload, "SIMT report")
    if not isinstance(report, dict):
        raise SimtError("SIMT report must be a JSON object")
    if payload != canonical_json_bytes(report) + b"\n":
        raise SimtError("SIMT report bytes are not canonical JSON")
    verify_simt_report(report)
    return report
