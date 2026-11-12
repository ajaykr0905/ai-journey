"""Canonical identities for reward tasks and ordered task sets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .verifiable_reward_contract import VerifiableRewardError
from .verifiable_reward_task import VerifiableTask


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VerifiableRewardError("value is not canonical finite JSON") from exc


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def task_set_identity(tasks: Iterable[VerifiableTask]) -> str:
    records = [task.to_mapping() for task in tasks]
    ids = [record["task_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise VerifiableRewardError("task identifiers must be unique")
    return sha256_json(records)
