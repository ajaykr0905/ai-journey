"""Content identities for Project 1 runs and attempts."""

from __future__ import annotations

import json
import re
from hashlib import sha256

from .project1_contract import Project1RunContract

_RUN_ID = re.compile(r"[a-z0-9-]+-[0-9a-f]{16}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def run_id(contract: Project1RunContract) -> str:
    """Return a readable identity bound to every run control."""

    if not isinstance(contract, Project1RunContract):
        raise TypeError("contract must be Project1RunContract")
    return f"{contract.run_name}-{contract.fingerprint()[:16]}"


def attempt_id(
    run_identity: str,
    attempt_number: int,
    *,
    parent_checkpoint_sha256: str | None = None,
) -> str:
    """Identify a fresh or resumed attempt and its declared parent state."""

    if not isinstance(run_identity, str) or _RUN_ID.fullmatch(run_identity) is None:
        raise ValueError("run_identity is invalid")
    if isinstance(attempt_number, bool) or not isinstance(attempt_number, int):
        raise TypeError("attempt_number must be an integer")
    if attempt_number <= 0:
        raise ValueError("attempt_number must be positive")
    if parent_checkpoint_sha256 is not None and (
        not isinstance(parent_checkpoint_sha256, str)
        or _DIGEST.fullmatch(parent_checkpoint_sha256) is None
    ):
        raise ValueError("parent checkpoint must be a SHA-256 digest")
    payload = {
        "run_id": run_identity,
        "attempt_number": attempt_number,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"attempt-{attempt_number}-{digest[:16]}"
