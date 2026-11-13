"""Deterministic result accumulation for recovery work items."""

from __future__ import annotations

import re
from hashlib import sha256

from .recovery_config import RecoveryExperimentError
from .recovery_workload import RecoveryWorkItem, RecoveryWorkload

_DIGEST = re.compile(r"[0-9a-f]{64}")
EMPTY_ACCUMULATOR_SHA256 = sha256(b"").hexdigest()


def accumulate_work_item(previous_sha256: str, item: RecoveryWorkItem) -> str:
    """Advance the result digest by one ordered work item."""

    if (
        not isinstance(previous_sha256, str)
        or _DIGEST.fullmatch(previous_sha256) is None
    ):
        raise RecoveryExperimentError("previous_sha256 must be a SHA-256 digest")
    if not isinstance(item, RecoveryWorkItem):
        raise TypeError("item must be RecoveryWorkItem")
    return sha256(
        bytes.fromhex(previous_sha256) + bytes.fromhex(item.payload_sha256)
    ).hexdigest()


def rebuild_accumulator(workload: RecoveryWorkload, completed_items: int) -> str:
    """Recompute a prefix result without trusting stored worker state."""

    if not isinstance(workload, RecoveryWorkload):
        raise TypeError("workload must be RecoveryWorkload")
    if isinstance(completed_items, bool) or not isinstance(completed_items, int):
        raise TypeError("completed_items must be an integer")
    if not 0 <= completed_items <= len(workload.items):
        raise RecoveryExperimentError("completed_items is outside the workload")
    digest = EMPTY_ACCUMULATOR_SHA256
    for item in workload.items[:completed_items]:
        digest = accumulate_work_item(digest, item)
    return digest
