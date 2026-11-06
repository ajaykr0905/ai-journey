"""Durable I/O for integrity-checked LAMBADA evaluation evidence."""

import json
import os
from pathlib import Path
import stat
import tempfile

from .lambada import LambadaError, verify_evidence


MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def _read_regular(path: Path, *, maximum: int) -> bytes:
    try:
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
                raise LambadaError("evidence input exceeds regular-file constraints")
            raw = stream.read(maximum + 1)
    except OSError as error:
        raise LambadaError("evidence input exceeds regular-file constraints") from error
    if len(raw) > maximum:
        raise LambadaError("evidence input exceeds its byte budget")
    return raw


def _parse_json(raw: bytes) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise LambadaError("evidence input contains duplicate JSON fields")
            result[key] = value
        return result

    def invalid_constant(value):
        raise LambadaError("evidence input must contain finite JSON numbers")

    try:
        payload = json.loads(
            raw, object_pairs_hook=unique_object, parse_constant=invalid_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise LambadaError("evidence input must be valid bounded UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise LambadaError("evidence input must contain a JSON object")
    return payload


def write_evidence(path: Path, payload: dict) -> None:
    """Atomically replace one validated report and flush its parent directory."""
    path = Path(path)
    verify_evidence(payload)
    if path.is_symlink():
        raise LambadaError("evidence output must not be a symlink")
    body = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if len(body) > MAX_EVIDENCE_BYTES:
        raise LambadaError("evidence output exceeds the file budget")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_evidence(path: Path) -> dict:
    """Read one bounded regular file before validating its complete payload."""
    return verify_evidence(
        _parse_json(_read_regular(Path(path), maximum=MAX_EVIDENCE_BYTES))
    )
