"""Strict bounded controls for the serving failure-injection recovery lab.

The lab kills a real loopback inference server under client load and measures
what clients observe. Every control is an explicit integer bound; missing,
unknown, duplicate and non-finite fields are rejected before a process starts.
This module imports only the standard library: the server child reuses its
JSON and file helpers and must cold-start without numpy or torch.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 16 * 1024
_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_DIGEST = re.compile(r"[0-9a-f]{64}")


class ServingRecoveryError(ValueError):
    """Raised when serving recovery inputs or evidence violate the contract."""


def require_int(value: object, name: str, minimum: int, maximum: int) -> int:
    """Return an exact integer inside an inclusive bound; reject booleans."""

    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ServingRecoveryError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def require_digest(value: object, name: str) -> str:
    """Return a lowercase hexadecimal SHA-256 digest."""

    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ServingRecoveryError(f"{name} must be a lowercase SHA-256 digest")
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Encode sorted, compact, ASCII-only JSON without non-finite numbers."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_hex(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact bytes."""

    if type(data) is not bytes:
        raise TypeError("digest input must be bytes")
    return sha256(data).hexdigest()


def canonical_sha256(value: object) -> str:
    """Digest the canonical JSON encoding of a value."""

    return sha256_hex(canonical_json_bytes(value))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ServingRecoveryError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(name: str) -> None:
    raise ServingRecoveryError(f"non-finite JSON constant: {name}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ServingRecoveryError("JSON number overflows a finite float")
    return value


def strict_json_loads(data: bytes, *, name: str) -> Any:
    """Decode strict UTF-8 JSON, rejecting duplicate keys and non-finite numbers."""

    if type(data) is not bytes:
        raise TypeError(f"{name} must be bytes")
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except ServingRecoveryError as exc:
        raise ServingRecoveryError(f"{name} is not strict JSON: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ServingRecoveryError(f"{name} is not strict UTF-8 JSON") from exc


def read_bounded_file(path: Path, *, max_bytes: int, name: str) -> bytes:
    """Read a regular non-symlink file whose size fits an explicit byte bound."""

    if not isinstance(path, Path):
        raise TypeError(f"{name} path must be pathlib.Path")
    require_int(max_bytes, "max_bytes", 1, 64 * 1024 * 1024)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ServingRecoveryError(
            f"{name} must be a readable regular non-symlink file"
        ) from exc
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ServingRecoveryError(f"{name} must be a regular file")
        if info.st_size > max_bytes:
            raise ServingRecoveryError(f"{name} exceeds its {max_bytes}-byte bound")
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if len(data) > max_bytes:
        raise ServingRecoveryError(f"{name} exceeds its {max_bytes}-byte bound")
    return data


_BOUNDS = {
    "seed": (0, 2**32 - 1),
    "requests": (2, 256),
    "concurrency": (1, 4),
    "max_new_tokens": (1, 32),
    "fault_after_acknowledged": (1, 255),
    "probe_interval_ms": (1, 1000),
    "probe_timeout_ms": (1, 5000),
    "detection_timeout_ms": (2, 10_000),
    "readiness_timeout_ms": (100, 30_000),
    "restart_limit": (1, 5),
    "restart_backoff_ms": (0, 5000),
    "request_deadline_ms": (10, 10_000),
    "retry_budget": (1, 64),
    "retry_backoff_ms": (1, 5000),
    "max_retry_backoff_ms": (1, 10_000),
    "slo_latency_ms": (1, 10_000),
    "slo_consecutive_successes": (1, 64),
    "max_run_ms": (1000, 300_000),
    "shutdown_timeout_ms": (100, 30_000),
}


@dataclass(frozen=True)
class ServingRecoveryConfig:
    """Bounded controls for one loopback serving kill-and-recover experiment.

    All durations are integer milliseconds. The fault fires when the client
    ledger has observed ``fault_after_acknowledged`` successful responses.
    """

    experiment_name: str
    seed: int
    requests: int
    concurrency: int
    max_new_tokens: int
    fault_after_acknowledged: int
    probe_interval_ms: int
    probe_timeout_ms: int
    detection_timeout_ms: int
    readiness_timeout_ms: int
    restart_limit: int
    restart_backoff_ms: int
    request_deadline_ms: int
    retry_budget: int
    retry_backoff_ms: int
    max_retry_backoff_ms: int
    slo_latency_ms: int
    slo_consecutive_successes: int
    max_run_ms: int
    shutdown_timeout_ms: int

    def __post_init__(self) -> None:
        if (
            type(self.experiment_name) is not str
            or _NAME.fullmatch(self.experiment_name) is None
        ):
            raise ServingRecoveryError("experiment_name must be a lowercase slug")
        for name, (minimum, maximum) in _BOUNDS.items():
            require_int(getattr(self, name), name, minimum, maximum)
        if self.concurrency > self.requests:
            raise ServingRecoveryError("concurrency cannot exceed requests")
        if self.fault_after_acknowledged >= self.requests:
            raise ServingRecoveryError(
                "fault_after_acknowledged must precede the final request"
            )
        if (
            self.requests
            < self.fault_after_acknowledged
            + self.concurrency
            - 1
            + self.slo_consecutive_successes
        ):
            raise ServingRecoveryError(
                "requests leave too few post-fault requests to observe SLO recovery"
            )
        if self.probe_interval_ms >= self.detection_timeout_ms:
            raise ServingRecoveryError(
                "probe_interval_ms must be lower than detection_timeout_ms"
            )
        if self.probe_timeout_ms > self.detection_timeout_ms:
            raise ServingRecoveryError(
                "probe_timeout_ms cannot exceed detection_timeout_ms"
            )
        if self.slo_latency_ms > self.request_deadline_ms:
            raise ServingRecoveryError(
                "slo_latency_ms cannot exceed request_deadline_ms"
            )
        if self.retry_backoff_ms > self.max_retry_backoff_ms:
            raise ServingRecoveryError(
                "retry_backoff_ms cannot exceed max_retry_backoff_ms"
            )
        if self.retry_window_ms < self.recovery_bound_ms:
            raise ServingRecoveryError(
                "retry backoff must outlast detection, restart backoff and readiness"
            )
        if (
            self.retry_budget + 1
        ) * self.request_deadline_ms + self.retry_window_ms > self.max_run_ms:
            raise ServingRecoveryError(
                "one request's worst-case retries exceed max_run_ms"
            )
        supervisor_bound = (
            (self.restart_limit + 1) * self.readiness_timeout_ms
            + sum(
                self.restart_delay_ms(restart)
                for restart in range(1, self.restart_limit + 1)
            )
            + self.detection_timeout_ms
            + self.shutdown_timeout_ms
        )
        if supervisor_bound > self.max_run_ms:
            raise ServingRecoveryError(
                "worst-case supervisor restarts and shutdown exceed max_run_ms"
            )

    def retry_delay_ms(self, retry: int) -> int:
        """Return the capped exponential backoff before retry number ``retry``."""

        require_int(retry, "retry", 1, self.retry_budget)
        return min(self.retry_backoff_ms * 2 ** (retry - 1), self.max_retry_backoff_ms)

    def restart_delay_ms(self, restart: int) -> int:
        """Return the doubling backoff before supervisor restart ``restart``."""

        require_int(restart, "restart", 1, self.restart_limit)
        return self.restart_backoff_ms * 2 ** (restart - 1)

    @property
    def retry_window_ms(self) -> int:
        """Total client backoff available across one request's retry budget."""

        return sum(
            min(self.retry_backoff_ms * 2**index, self.max_retry_backoff_ms)
            for index in range(self.retry_budget)
        )

    @property
    def recovery_bound_ms(self) -> int:
        """Worst admitted outage for one restart: detect, back off, become ready."""

        return (
            self.detection_timeout_ms
            + self.restart_backoff_ms
            + self.readiness_timeout_ms
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical versioned mapping."""

        return {
            "schema_version": SCHEMA_VERSION,
            **{field.name: getattr(self, field.name) for field in fields(self)},
        }

    def fingerprint(self) -> str:
        """Bind every control to one stable digest."""

        return canonical_sha256(self.to_mapping())


_FIELDS = frozenset(
    {"schema_version"} | {f.name for f in fields(ServingRecoveryConfig)}
)


def config_from_mapping(payload: object) -> ServingRecoveryConfig:
    """Parse controls without ignoring missing or unknown fields."""

    if type(payload) is not dict or set(payload) != _FIELDS:
        raise ServingRecoveryError("serving recovery config fields do not match schema")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != SCHEMA_VERSION
    ):
        raise ServingRecoveryError("serving recovery config schema is unsupported")
    return ServingRecoveryConfig(
        **{key: value for key, value in payload.items() if key != "schema_version"}
    )


def load_serving_recovery_config(path: Path) -> ServingRecoveryConfig:
    """Load bounded strict JSON controls from a regular file."""

    data = read_bounded_file(path, max_bytes=MAX_CONFIG_BYTES, name="config")
    return config_from_mapping(strict_json_loads(data, name="config"))
