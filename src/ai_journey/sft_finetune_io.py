"""Atomic bounded JSON storage for sealed SFT reports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .sft_finetune_contract import SFTFineTuneError
from .sft_finetune_report import validate_finetune_report

REPORT_BYTES = 8 * 1024 * 1024


def write_finetune_report(path: Path, report: dict) -> None:
    validate_finetune_report(report)
    encoded = (
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > REPORT_BYTES:
        raise SFTFineTuneError("report exceeds the eight MiB storage bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SFTFineTuneError("report path must be a regular non-symlink file")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_finetune_report(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > REPORT_BYTES:
        raise SFTFineTuneError("report must be a bounded regular non-symlink file")
    try:
        return validate_finetune_report(json.loads(path.read_bytes()))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SFTFineTuneError("report is not valid UTF-8 JSON") from exc
