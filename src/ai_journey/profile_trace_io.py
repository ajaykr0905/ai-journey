"""Publish and load one capture directory: a sealed report beside its traces.

A capture directory holds ``report.json`` and one trace file per shape,
under the plain names that the layout module allows. Saving verifies the
report against the trace bytes in a scratch directory first, so no file
this lab writes can fail its own verifier. Each file is then written to a
fsynced temporary file in the target directory and moved into place with
``os.replace``: the traces first, the report last, so a reader never finds
a report whose traces are missing. An output that aliases a protected
input, such as the config, is refused.

Loading reads the report with the shared bounded reader, verifies it
against the traces in its own directory, and requires canonical bytes.
Unreadable or malformed files raise ``ProfileTraceError``; a report that
was read but fails verification raises ``ProfileTraceVerificationError``.
Nothing here imports torch.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from .memory_access_pattern import MemoryAccessError
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import (
    MAX_REPORT_BYTES,
    parse_json_object,
    paths_alias,
    read_bounded_bytes,
    write_report,
)
from .profile_trace_capture import CapturedShape
from .profile_trace_config import ProfileTraceConfig
from .profile_trace_events import ProfileTraceError
from .profile_trace_layout import require_trace_file_name
from .profile_trace_verify import ProfileTraceVerificationError, verify_report

REPORT_FILE = "report.json"


def _publish(path: Path, payload: bytes) -> None:
    """Write bytes atomically next to their final path."""

    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ProfileTraceError(f"{path.name} must be a regular non-symlink file")
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


def output_paths(directory: Path, captured: tuple[CapturedShape, ...]) -> list[Path]:
    """Return every path a save would write, the report last."""

    if not isinstance(directory, Path):
        raise TypeError("output directory must be pathlib.Path")
    names = [require_trace_file_name(item.file_name) for item in captured]
    return [directory / name for name in names] + [directory / REPORT_FILE]


def save_capture(
    directory: Path,
    report: dict[str, object],
    captured: tuple[CapturedShape, ...],
    *,
    config: ProfileTraceConfig | None = None,
    protected: Iterable[Path] = (),
) -> dict[str, object]:
    """Verify, then publish the traces and the report; return the summary."""

    paths = output_paths(directory, captured)
    for path in paths:
        for item in protected:
            if paths_alias(path, item):
                raise ProfileTraceError(
                    "capture output must not alias a protected input"
                )
    with tempfile.TemporaryDirectory(prefix="profile-trace-verify-") as scratch:
        for item in captured:
            (Path(scratch) / item.file_name).write_bytes(item.trace_bytes)
        summary = verify_report(report, trace_dir=Path(scratch), config=config)
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ProfileTraceError("capture output must be a real directory")
    directory.mkdir(parents=True, exist_ok=True)
    for path, item in zip(paths, captured):
        _publish(path, item.trace_bytes)
    try:
        write_report(paths[-1], report, protected=tuple(protected))
    except MemoryAccessError as error:
        raise ProfileTraceError(str(error)) from error
    return summary


def load_report(
    path: Path, *, config: ProfileTraceConfig | None = None
) -> tuple[dict[str, object], dict[str, object]]:
    """Read and verify one report against the traces beside it.

    Verification runs before the canonical-bytes check, so a tampered report
    is named as such even when it was also reformatted.
    """

    try:
        payload = read_bounded_bytes(path, max_bytes=MAX_REPORT_BYTES, label="report")
        report = parse_json_object(payload, label="report")
    except MemoryAccessError as error:
        raise ProfileTraceError(str(error)) from error
    summary = verify_report(report, trace_dir=path.parent, config=config)
    if payload != canonical_json_bytes(report) + b"\n":
        raise ProfileTraceVerificationError("report file is not canonical JSON")
    return report, summary
