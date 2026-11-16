"""Canonical JSON, strict bounded reads and atomic writes for the launch lab.

Plans and reports are canonical JSON: sorted keys, compact separators, ASCII
escapes, and no NaN or infinity. A report digest is the SHA-256 of that
encoding. It detects drift and is not a signature. Reads open only regular
files without following a final symlink, stop at a byte bound, and decode
strict UTF-8 JSON that rejects duplicate keys, non-finite numbers and integer
literals too long to parse. Writes go through a fsynced temporary file in the
target directory and ``os.replace``, and refuse a target that aliases a
protected input such as the plan. Only the standard library is imported, so
the lab runs without torch or NumPy.
"""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from .kernel_launch_config import KernelLaunchError

MAX_FILE_BYTES = 4 * 1024 * 1024


def canonical_json_bytes(value: object) -> bytes:
    """Encode sorted, compact, ASCII-only JSON that has no NaN or infinity."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as error:
        raise KernelLaunchError("value cannot be encoded as canonical JSON") from error


def canonical_sha256(value: object) -> str:
    """Return the SHA-256 of a value's canonical JSON bytes."""

    return sha256(canonical_json_bytes(value)).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise KernelLaunchError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(name: str) -> object:
    raise KernelLaunchError(f"non-finite JSON constant: {name}")


def _finite_float(literal: str) -> float:
    value = float(literal)
    if not math.isfinite(value):
        raise KernelLaunchError(f"non-finite JSON number: {literal}")
    return value


def loads_strict(payload: bytes, *, label: str) -> object:
    """Decode UTF-8 JSON that has unique keys and finite numbers only."""

    if type(payload) is not bytes:
        raise TypeError(f"{label} must be bytes")
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except KernelLaunchError as error:
        raise KernelLaunchError(f"{label} is not strict JSON: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise KernelLaunchError(f"{label} is not strict UTF-8 JSON") from error
    except ValueError as error:
        # int() refuses literals longer than sys.get_int_max_str_digits().
        raise KernelLaunchError(
            f"{label} holds an integer literal too long to parse"
        ) from error


def read_bounded(path: Path, *, limit: int, label: str) -> bytes:
    """Read one regular, non-symlink file of at most ``limit`` bytes.

    ``O_NONBLOCK`` keeps a FIFO from blocking the open, and the file type is
    checked on the open descriptor, so a swapped path cannot slip through.
    """

    if not isinstance(path, Path):
        raise TypeError(f"{label} path must be pathlib.Path")
    if type(limit) is not int or not 1 <= limit <= MAX_FILE_BYTES:
        raise KernelLaunchError(f"limit must be an integer in [1, {MAX_FILE_BYTES}]")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise KernelLaunchError(
            f"{label} must be a readable regular non-symlink file"
        ) from error
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise KernelLaunchError(f"{label} must be a regular file")
        if info.st_size > limit:
            raise KernelLaunchError(f"{label} exceeds {limit} bytes")
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) > limit:
        raise KernelLaunchError(f"{label} exceeds {limit} bytes")
    return payload


def paths_alias(first: Path, second: Path) -> bool:
    """Return whether two paths name one file, through symlinks or hard links.

    ``os.path.realpath`` resolves as far as it can without raising, so a
    symlink loop compares like any other path instead of crashing, as
    ``Path.resolve`` does on Python 3.11.
    """

    if os.path.realpath(first) == os.path.realpath(second):
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


def write_atomic(path: Path, payload: bytes, *, protected: Iterable[Path] = ()) -> None:
    """Replace ``path`` with ``payload`` in one step, never over a protected file.

    A reader sees either the old file or the complete new one. The target
    must be a regular file or absent, and must not resolve to, or be a hard
    link of, any protected input.
    """

    if not isinstance(path, Path):
        raise TypeError("output path must be pathlib.Path")
    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    if path.name in ("", ".", ".."):
        raise KernelLaunchError("output path must name a file")
    for item in protected:
        if not isinstance(item, Path):
            raise TypeError("protected paths must be pathlib.Path")
        if paths_alias(path, item):
            raise KernelLaunchError("output must not alias a protected input")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise KernelLaunchError("output must be a regular non-symlink file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
