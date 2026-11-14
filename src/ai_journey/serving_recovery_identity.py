"""Run, server-lifetime and idempotency identities for serving recovery runs.

Identities are derived rather than random, so a reviewer can recompute every
run, lifetime and request key from the sealed config and model digests. A
request keeps one idempotency key across all attempts and server lifetimes.
The module imports only the standard library because the server validates keys.
"""

from __future__ import annotations

import re

from .serving_recovery_config import (
    ServingRecoveryConfig,
    ServingRecoveryError,
    canonical_sha256,
    require_digest,
    require_int,
)

MAX_LIFETIMES = 64
MAX_REQUEST_INDEX = 9999
_RUN_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?-[0-9a-f]{16}")
_LIFETIME_ID = re.compile(r"lifetime-([1-9][0-9]?)-[0-9a-f]{16}")
_KEY = re.compile(r"req-([0-9]{4})-[0-9a-f]{24}")


def serving_run_id(config: ServingRecoveryConfig, *, model_sha256: str) -> str:
    """Return a readable run identity bound to the config and model digests."""

    if not isinstance(config, ServingRecoveryConfig):
        raise TypeError("config must be ServingRecoveryConfig")
    require_digest(model_sha256, "model_sha256")
    digest = canonical_sha256(
        {
            "kind": "serving-recovery-run",
            "config_sha256": config.fingerprint(),
            "model_sha256": model_sha256,
        }
    )
    return f"{config.experiment_name}-{digest[:16]}"


def require_run_id(value: object) -> str:
    """Return a syntactically valid run identity."""

    if type(value) is not str or _RUN_ID.fullmatch(value) is None:
        raise ServingRecoveryError("run_id is invalid")
    return value


def server_lifetime_id(run_id: str, lifetime_number: int) -> str:
    """Identify the n-th supervised server process of one run."""

    require_run_id(run_id)
    require_int(lifetime_number, "lifetime_number", 1, MAX_LIFETIMES)
    digest = canonical_sha256(
        {
            "kind": "serving-recovery-lifetime",
            "run_id": run_id,
            "lifetime_number": lifetime_number,
        }
    )
    return f"lifetime-{lifetime_number}-{digest[:16]}"


def require_lifetime_id(value: object) -> str:
    """Return a syntactically valid server lifetime identity."""

    if type(value) is not str or _LIFETIME_ID.fullmatch(value) is None:
        raise ServingRecoveryError("lifetime_id is invalid")
    if not 1 <= int(_LIFETIME_ID.fullmatch(value).group(1)) <= MAX_LIFETIMES:
        raise ServingRecoveryError("lifetime_id number is out of range")
    return value


def lifetime_number(lifetime_id: str) -> int:
    """Return the 1-based process number embedded in a lifetime identity."""

    require_lifetime_id(lifetime_id)
    return int(_LIFETIME_ID.fullmatch(lifetime_id).group(1))


def idempotency_key(run_id: str, request_index: int) -> str:
    """Return the stable key every attempt of one logical request carries."""

    require_run_id(run_id)
    require_int(request_index, "request_index", 0, MAX_REQUEST_INDEX)
    digest = canonical_sha256(
        {
            "kind": "serving-recovery-request",
            "run_id": run_id,
            "request_index": request_index,
        }
    )
    return f"req-{request_index:04d}-{digest[:24]}"


def require_idempotency_key(value: object) -> str:
    """Return a syntactically valid idempotency key."""

    if type(value) is not str or _KEY.fullmatch(value) is None:
        raise ServingRecoveryError("idempotency key is invalid")
    return value


def key_request_index(key: str) -> int:
    """Return the logical request index embedded in an idempotency key."""

    require_idempotency_key(key)
    return int(_KEY.fullmatch(key).group(1))


def require_run_key(run_id: str, key: str) -> int:
    """Return the request index of a key only if this run derived it."""

    index = key_request_index(key)
    if idempotency_key(run_id, index) != key:
        raise ServingRecoveryError("idempotency key does not belong to this run")
    return index
