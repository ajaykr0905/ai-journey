"""Durable I/O for integrity-checked HellaSwag benchmark evidence."""

import json
import os
from pathlib import Path
import tempfile

from .hellaswag import HellaSwagError, verify_evidence
from .hellaswag_io import read_evaluation_bytes, parse_evaluation_json


MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def write_evidence(path: Path, payload: dict) -> None:
    """Atomically replace one validated report and flush its parent directory."""
    path = Path(path)
    verify_evidence(payload)
    if path.is_symlink():
        raise HellaSwagError("evidence output must not be a symlink")
    body = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if len(body) > MAX_EVIDENCE_BYTES:
        raise HellaSwagError("evidence output exceeds the file budget")
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
    path = Path(path)
    payload = parse_evaluation_json(
        read_evaluation_bytes(path, maximum=MAX_EVIDENCE_BYTES, name="evidence input"),
        name="evidence input",
    )
    return verify_evidence(payload)
