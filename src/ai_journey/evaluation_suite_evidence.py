"""Durable I/O for verified three-axis evaluation suites."""

import json
import os
from pathlib import Path
import stat
import tempfile

from .evaluation_suite import EvaluationSuiteError, verify_evaluation_suite


MAX_SUITE_BYTES = 128 * 1024


def _read_regular(path: Path) -> bytes:
    try:
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or not 0 < info.st_size <= MAX_SUITE_BYTES
            ):
                raise EvaluationSuiteError(
                    "suite input exceeds regular-file constraints"
                )
            raw = stream.read(MAX_SUITE_BYTES + 1)
    except OSError as error:
        raise EvaluationSuiteError(
            "suite input exceeds regular-file constraints"
        ) from error
    if len(raw) > MAX_SUITE_BYTES:
        raise EvaluationSuiteError("suite input exceeds its byte budget")
    return raw


def _parse_json(raw: bytes) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationSuiteError("suite input contains duplicate JSON fields")
            result[key] = value
        return result

    def invalid_constant(value):
        raise EvaluationSuiteError("suite input must contain finite JSON numbers")

    try:
        payload = json.loads(
            raw, object_pairs_hook=unique_object, parse_constant=invalid_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise EvaluationSuiteError(
            "suite input must be valid bounded UTF-8 JSON"
        ) from error
    if not isinstance(payload, dict):
        raise EvaluationSuiteError("suite input must contain a JSON object")
    return payload


def write_suite(path: Path, payload: dict) -> None:
    """Atomically replace one verified suite and flush its parent directory."""
    path = Path(path)
    verify_evaluation_suite(payload)
    if path.is_symlink():
        raise EvaluationSuiteError("suite output must not be a symlink")
    body = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if len(body) > MAX_SUITE_BYTES:
        raise EvaluationSuiteError("suite output exceeds the file budget")
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


def read_suite(path: Path) -> dict:
    """Read and verify one bounded regular suite file."""
    return verify_evaluation_suite(_parse_json(_read_regular(Path(path))))
