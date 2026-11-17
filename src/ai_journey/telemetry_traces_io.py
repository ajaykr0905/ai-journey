"""Strict canonical JSON, bounded reads and no-overwrite publication for telemetry.

Telemetry artifacts are derived evidence: an OTLP/JSON trace export, a Grafana
dashboard and the sealed reports that bind them to their sources. A report's
seal is the SHA-256 of its canonical JSON without the seal field; it proves
content identity, not authorship. Readers open
only regular files without following a final symlink, stop at an explicit byte
bound and decode strict UTF-8 JSON that rejects duplicate keys, non-finite
numbers and integer literals too long to parse. Writers publish complete bytes
through a fsynced temporary file hard-linked into a path that does not exist
yet; unlike ``os.replace``, ``os.link`` fails if the destination appeared, so
no artifact or source file is ever overwritten. Only the standard library is
imported, so the telemetry lab runs without torch or NumPy.
"""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024


class TelemetryError(ValueError):
    """Raised when telemetry inputs, artifacts or reports break their contract."""


class TelemetryVerificationError(TelemetryError):
    """Raised when source evidence or a sealed telemetry artifact is rejected."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode sorted, compact, ASCII-only JSON without non-finite numbers."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as error:
        raise TelemetryError("value cannot be encoded as canonical JSON") from error


def artifact_bytes(value: object) -> bytes:
    """Return the published file form: canonical JSON plus one newline."""

    return canonical_json_bytes(value) + b"\n"


def sha256_hex(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact bytes."""

    if type(data) is not bytes:
        raise TypeError("digest input must be bytes")
    return sha256(data).hexdigest()


def canonical_sha256(value: object) -> str:
    """Digest the canonical JSON encoding of a value (no trailing newline)."""

    return sha256_hex(canonical_json_bytes(value))


def seal_report(report: dict[str, object]) -> dict[str, object]:
    """Add the canonical SHA-256 seal over every other report field."""

    if type(report) is not dict or "report_sha256" in report:
        raise TelemetryError("report must be an unsealed mapping")
    return {**report, "report_sha256": canonical_sha256(report)}


def require_sealed_report(report: object, *, kind: str) -> dict[str, object]:
    """Return a report of ``kind`` whose seal matches its canonical content."""

    if type(report) is not dict or type(report.get("report_sha256")) is not str:
        raise TelemetryError("report is not sealed")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if canonical_sha256(unsigned) != report["report_sha256"]:
        raise TelemetryError("report seal does not match its content")
    if report.get("kind") != kind:
        raise TelemetryError(f"report kind is not {kind}")
    return report


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise TelemetryError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(name: str) -> object:
    raise TelemetryError(f"non-finite JSON constant: {name}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise TelemetryError("JSON number overflows a finite float")
    return value


def strict_json_loads(data: bytes, *, label: str) -> object:
    """Decode strict UTF-8 JSON with unique keys and finite numbers only."""

    if type(data) is not bytes:
        raise TypeError(f"{label} must be bytes")
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except TelemetryError as error:
        raise TelemetryError(f"{label} is not strict JSON: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise TelemetryError(f"{label} is not strict UTF-8 JSON") from error
    except ValueError as error:
        # int() refuses literals longer than sys.get_int_max_str_digits().
        raise TelemetryError(
            f"{label} holds an integer literal too long to parse"
        ) from error


def _require_bound(max_bytes: object) -> int:
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_ARTIFACT_BYTES:
        raise TelemetryError(
            f"max_bytes must be an integer in [1, {MAX_ARTIFACT_BYTES}]"
        )
    return max_bytes


def read_bounded_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    """Read a regular non-symlink file whose size fits an explicit byte bound."""

    if not isinstance(path, Path):
        raise TypeError(f"{label} path must be pathlib.Path")
    _require_bound(max_bytes)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TelemetryError(
            f"{label} must be a readable regular non-symlink file"
        ) from error
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise TelemetryError(f"{label} must be a regular file")
        if info.st_size > max_bytes:
            raise TelemetryError(f"{label} exceeds its {max_bytes}-byte bound")
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if len(data) > max_bytes:
        raise TelemetryError(f"{label} exceeds its {max_bytes}-byte bound")
    return data


def load_canonical_artifact(path: Path, *, max_bytes: int, label: str) -> object:
    """Load a bounded artifact whose bytes are exactly its published form."""

    data = read_bounded_file(path, max_bytes=max_bytes, label=label)
    value = strict_json_loads(data, label=label)
    if artifact_bytes(value) != data:
        raise TelemetryError(f"{label} is not canonical JSON with one final newline")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_new_file(path: Path, data: bytes, *, max_bytes: int) -> None:
    """Atomically publish complete bytes at a path that does not exist yet."""

    if not isinstance(path, Path):
        raise TypeError("artifact path must be pathlib.Path")
    if type(data) is not bytes:
        raise TypeError("artifact data must be bytes")
    _require_bound(max_bytes)
    if len(data) > max_bytes:
        raise TelemetryError(f"{path.name} exceeds its {max_bytes}-byte bound")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise TelemetryError("artifact parent must be a real directory")
    if os.path.lexists(path):
        raise TelemetryError(f"refusing to overwrite {path.name}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise TelemetryError(f"refusing to overwrite {path.name}") from error
        _fsync_directory(parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_output_directory(path: Path, *, sources: Sequence[Path] = ()) -> Path:
    """Create a new artifact directory outside every source evidence directory.

    Derived telemetry must never be written into the run directories it was
    derived from, so a path equal to or inside any source is refused.
    """

    if not isinstance(path, Path):
        raise TypeError("output directory must be pathlib.Path")
    if any(not isinstance(source, Path) for source in sources):
        raise TypeError("source directories must be pathlib.Path values")
    if os.path.lexists(path):
        raise TelemetryError("output directory must not already exist")
    try:
        target = path.resolve()
        roots = [source.resolve() for source in sources]
    except (OSError, RuntimeError) as error:
        # Python 3.11 raises RuntimeError when a path is a symlink loop.
        raise TelemetryError("output and source paths must resolve") from error
    if any(target == root or target.is_relative_to(root) for root in roots):
        raise TelemetryError("output directory must not be inside a source directory")
    path.mkdir(parents=True)
    return path
