"""Strict canonical JSON and bounded file reads for SIMT plans and reports."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat

from .simt_device import SimtError


def canonical_json_bytes(value: object) -> bytes:
    """Return sorted, compact, ASCII-only JSON that rejects NaN and infinity."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as error:
        raise SimtError("value cannot be encoded as canonical JSON") from error


def canonical_sha256(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SimtError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(name: str) -> object:
    raise SimtError(f"non-finite JSON number: {name}")


def loads_strict(payload: bytes, label: str) -> object:
    """Parse UTF-8 JSON, rejecting duplicate fields and non-finite numbers."""

    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except SimtError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise SimtError(f"{label} is not strict UTF-8 JSON") from error


def read_bounded(path: Path, *, limit: int, label: str) -> bytes:
    """Read one regular, non-symlink file of at most ``limit`` bytes."""

    if not isinstance(path, Path):
        raise TypeError(f"{label} path must be pathlib.Path")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SimtError(f"unable to read {label}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise SimtError(f"{label} must be a regular non-symlink file")
    if metadata.st_size > limit:
        raise SimtError(f"{label} exceeds {limit} bytes")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SimtError(f"unable to read {label}") from error
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        with stream:
            payload = stream.read(limit + 1)
    except OSError as error:
        raise SimtError(f"unable to read {label}") from error
    if len(payload) > limit:
        raise SimtError(f"{label} exceeds {limit} bytes")
    return payload
