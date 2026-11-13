"""Bounded strict JSON input and atomic publication of sealed reports.

Reads open only regular files without following a final symlink, stop at a
byte bound, decode strict UTF-8 JSON, reject duplicate keys and non-finite
constants, and require an object root. Reports must also be byte-for-byte
canonical and carry a matching digest. Writes accept only sealed reports and
publish canonical JSON atomically: a temporary file in the target directory is
written, flushed and fsynced, then moved into place with ``os.replace``. An
output path that aliases a protected input is refused.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Iterable
from pathlib import Path

from .memory_access_pattern import MemoryAccessError
from .memory_access_plan import (
    MemoryAccessPlan,
    canonical_json_bytes,
    plan_from_mapping,
)
from .memory_access_report import DIGEST_FIELD, report_digest

MAX_PLAN_BYTES = 64 * 1024
MAX_REPORT_BYTES = 1024 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MemoryAccessError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(name: str) -> object:
    raise MemoryAccessError(f"non-finite JSON constant: {name}")


def read_bounded_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    """Read at most ``max_bytes`` from a regular file without following a link."""

    if not isinstance(path, Path):
        raise TypeError(f"{label} path must be pathlib.Path")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise MemoryAccessError("max_bytes must be a positive integer")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MemoryAccessError(
            f"{label} must be a readable regular non-symlink file"
        ) from error
    try:
        regular = stat.S_ISREG(os.fstat(descriptor).st_mode)
    except OSError:
        regular = False
    if not regular:
        os.close(descriptor)
        raise MemoryAccessError(f"{label} must be a regular non-symlink file")
    with os.fdopen(descriptor, "rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise MemoryAccessError(f"{label} exceeds {max_bytes} bytes")
    return payload


def parse_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    """Decode strict UTF-8 JSON with unique keys, finite numbers, object root."""

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except MemoryAccessError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise MemoryAccessError(f"{label} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise MemoryAccessError(f"{label} root must be a JSON object")
    return value


def read_json_object(path: Path, *, max_bytes: int, label: str) -> dict[str, object]:
    """Read one bounded strict JSON object from a regular file."""

    payload = read_bounded_bytes(path, max_bytes=max_bytes, label=label)
    return parse_json_object(payload, label=label)


def load_plan(path: Path) -> MemoryAccessPlan:
    """Load and validate one bounded plan file."""

    return plan_from_mapping(
        read_json_object(path, max_bytes=MAX_PLAN_BYTES, label="plan")
    )


def load_report(path: Path) -> dict[str, object]:
    """Load a canonical sealed report; deep checks belong to the verifier."""

    payload = read_bounded_bytes(path, max_bytes=MAX_REPORT_BYTES, label="report")
    report = parse_json_object(payload, label="report")
    if report.get(DIGEST_FIELD) != report_digest(report):
        raise MemoryAccessError("report digest does not match its content")
    if payload != canonical_json_bytes(report) + b"\n":
        raise MemoryAccessError("report file is not canonical JSON")
    return report


def paths_alias(first: Path, second: Path) -> bool:
    """Return whether two paths name the same file, including links.

    ``os.path.realpath`` resolves as far as it can without raising, so a
    symlink loop is compared like any other path instead of crashing (Python
    3.11's ``Path.resolve`` raises RuntimeError); the strict reader and the
    writer then refuse the loop itself.
    """

    if os.path.realpath(first) == os.path.realpath(second):
        return True
    try:
        return first.exists() and second.exists() and first.samefile(second)
    except OSError:
        return False


def write_report(
    path: Path, report: dict[str, object], *, protected: Iterable[Path] = ()
) -> None:
    """Atomically publish one sealed canonical report."""

    if not isinstance(path, Path):
        raise TypeError("report path must be pathlib.Path")
    if type(report) is not dict or report.get(DIGEST_FIELD) != report_digest(report):
        raise MemoryAccessError(
            "only sealed reports with a matching digest are written"
        )
    payload = canonical_json_bytes(report) + b"\n"
    if len(payload) > MAX_REPORT_BYTES:
        raise MemoryAccessError(f"report exceeds {MAX_REPORT_BYTES} bytes")
    for item in protected:
        if not isinstance(item, Path):
            raise TypeError("protected paths must be pathlib.Path")
        if paths_alias(path, item):
            raise MemoryAccessError("report output must not alias a protected input")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise MemoryAccessError("report output must be a regular non-symlink file")
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
