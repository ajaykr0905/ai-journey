"""Strict bounded loading and atomic publication of reward reports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .verifiable_reward_contract import VerifiableRewardError
from .verifiable_reward_identity import canonical_json_bytes


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise VerifiableRewardError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def load_reward_report(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
        raise VerifiableRewardError("report must be a bounded regular non-symlink file")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise VerifiableRewardError("report is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise VerifiableRewardError("report root must be an object")
    return value


def write_reward_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
