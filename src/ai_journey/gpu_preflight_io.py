"""Bounded JSON records for NVIDIA preflight observations."""

import json
import os
from pathlib import Path
import stat
import tempfile

MAX_GPU_RECORD_BYTES = 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate preflight JSON field")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("preflight JSON numbers must be finite")


def read_gpu_json(path: Path) -> dict:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= MAX_GPU_RECORD_BYTES
        ):
            raise ValueError("preflight JSON exceeds regular-file byte constraints")
        raw = stream.read(MAX_GPU_RECORD_BYTES + 1)
    if len(raw) > MAX_GPU_RECORD_BYTES:
        raise ValueError("preflight JSON exceeds its byte budget")
    payload = json.loads(
        raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
    )
    if not isinstance(payload, dict):
        raise ValueError("preflight JSON must contain an object")
    return payload


def write_gpu_json(path: Path, payload: dict):
    raw = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if len(raw) > MAX_GPU_RECORD_BYTES:
        raise ValueError("preflight JSON exceeds its byte budget")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("preflight output must be a regular non-symlink path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
