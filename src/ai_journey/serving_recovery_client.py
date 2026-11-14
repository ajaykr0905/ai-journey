"""Bounded loopback generation attempts with exhaustive outcome classes.

Each attempt derives every socket timeout from one absolute monotonic deadline
and ends in exactly one class: ``ok``, ``connection_refused`` (the connect was
refused, ECONNREFUSED), ``connect_error`` (any other connect error, such as a
local EMFILE or EADDRNOTAVAIL, which says nothing about the server),
``connection_reset`` (including a peer close before a complete response),
``timeout`` (including a connect that ran out of time, ``connect_timeout``),
``http_error`` (a complete non-200 response) or ``invalid_response`` (a
complete 200 that breaks the protocol). A record keeps
monotonic start and finish times, the lifetime that answered and the SHA-256
of the exact response body, so later reconciliation can byte-compare retries.
"""

from __future__ import annotations

import errno
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

from .http_measurements import loopback_address
from .serving_recovery_config import (
    ServingRecoveryConfig,
    ServingRecoveryError,
    canonical_json_bytes,
    require_digest,
    require_int,
    sha256_hex,
    strict_json_loads,
)
from .serving_recovery_identity import (
    idempotency_key,
    key_request_index,
    require_lifetime_id,
)
from .serving_recovery_journal import MAX_ATTEMPT
from .serving_recovery_model import GenerationRequest, ServingModel
from .serving_recovery_server import ATTEMPT_HEADER, KEY_HEADER, LIFETIME_HEADER

ATTEMPT_OUTCOMES = (
    "ok",
    "connection_refused",
    "connect_error",
    "connection_reset",
    "timeout",
    "http_error",
    "invalid_response",
)
MAX_RESPONSE_BYTES = 16 * 1024
_STATUS = re.compile(rb"HTTP/1\.[01] ([1-5][0-9]{2})(?: [ -~]*)?")
_LENGTH = re.compile(r"0|[1-9][0-9]{0,5}")
_RESPONSE_FIELDS = frozenset(
    {"finish_reason", "max_new_tokens", "model_sha256", "output", "prompt", "seed"}
)


@dataclass(frozen=True)
class PlannedRequest:
    """One logical request and the idempotency key all its attempts carry."""

    index: int
    idempotency_key: str
    request: GenerationRequest

    def __post_init__(self) -> None:
        require_int(self.index, "index", 0, 9999)
        if key_request_index(self.idempotency_key) != self.index:
            raise ServingRecoveryError("idempotency key names another request")
        if not isinstance(self.request, GenerationRequest):
            raise TypeError("request must be GenerationRequest")


def plan_requests(
    config: ServingRecoveryConfig, *, run_id: str, model: ServingModel
) -> tuple[PlannedRequest, ...]:
    """Derive the deterministic prompts, seeds and keys of one run."""

    if not isinstance(config, ServingRecoveryConfig):
        raise TypeError("config must be ServingRecoveryConfig")
    if not isinstance(model, ServingModel):
        raise TypeError("model must be ServingModel")
    letters = model.vocabulary[1:]
    planned = []
    for index in range(config.requests):
        prompt = "".join(
            letters[(5 * index + 3 * position) % len(letters)]
            for position in range(1 + index % 3)
        )
        request = GenerationRequest(
            prompt, (config.seed + index) % 2**32, config.max_new_tokens
        )
        planned.append(PlannedRequest(index, idempotency_key(run_id, index), request))
    return tuple(planned)


@dataclass(frozen=True)
class AttemptRecord:
    """One classified client attempt on the monotonic clock."""

    request_index: int
    idempotency_key: str
    attempt: int
    port: int
    started_ns: int
    finished_ns: int
    outcome: str
    error: str | None
    http_status: int | None
    lifetime_id: str | None
    response_sha256: str | None

    def __post_init__(self) -> None:
        require_int(self.request_index, "request_index", 0, 9999)
        if key_request_index(self.idempotency_key) != self.request_index:
            raise ServingRecoveryError("attempt key names another request")
        require_int(self.attempt, "attempt", 1, MAX_ATTEMPT)
        require_int(self.port, "port", 1, 65535)
        require_int(self.started_ns, "started_ns", 0, 2**63 - 1)
        require_int(self.finished_ns, "finished_ns", self.started_ns, 2**63 - 1)
        if self.outcome not in ATTEMPT_OUTCOMES:
            raise ServingRecoveryError("attempt outcome is unsupported")
        if self.http_status is not None:
            require_int(self.http_status, "http_status", 100, 599)
        if self.outcome == "ok":
            require_lifetime_id(self.lifetime_id)
            require_digest(self.response_sha256, "response_sha256")
            if self.http_status != 200 or self.error is not None:
                raise ServingRecoveryError("ok attempts are clean 200 responses")
            return
        if self.lifetime_id is not None or self.response_sha256 is not None:
            raise ServingRecoveryError("failed attempts carry no served identity")
        if (
            type(self.error) is not str
            or re.fullmatch(r"[A-Za-z0-9_]{1,64}", self.error) is None
        ):
            raise ServingRecoveryError("failed attempts need a short error code")
        if self.outcome == "http_error" and self.http_status in (None, 200):
            raise ServingRecoveryError("http_error attempts need a non-200 status")

    @property
    def latency_ns(self) -> int:
        return self.finished_ns - self.started_ns

    @property
    def connected(self) -> bool:
        """False when the connect itself failed, so no server could run it."""

        return self.outcome not in ("connection_refused", "connect_error") and (
            self.outcome,
            self.error,
        ) != ("timeout", "connect_timeout")

    def to_mapping(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


_ATTEMPT_FIELDS = frozenset(field.name for field in fields(AttemptRecord))


def attempt_from_mapping(payload: object) -> AttemptRecord:
    """Parse one exact attempt record mapping."""

    if type(payload) is not dict or set(payload) != _ATTEMPT_FIELDS:
        raise ServingRecoveryError("attempt record fields do not match schema")
    return AttemptRecord(**payload)


def _remaining(deadline_ns: int, clock_ns: Callable[[], int]) -> float:
    remaining = (deadline_ns - clock_ns()) / 1e9
    if remaining <= 0:
        raise TimeoutError("attempt deadline elapsed")
    return remaining


def _exchange(
    port: int, wire: bytes, deadline_ns: int, clock_ns: Callable[[], int]
) -> tuple[str, str | None, bytes]:
    try:
        connection = socket.create_connection(
            ("127.0.0.1", port), timeout=_remaining(deadline_ns, clock_ns)
        )
    except TimeoutError:
        return "timeout", "connect_timeout", b""
    except ConnectionRefusedError as exc:
        return "connection_refused", type(exc).__name__, b""
    except OSError as exc:
        return "connect_error", errno.errorcode.get(exc.errno, type(exc).__name__), b""
    chunks: list[bytes] = []
    size = 0
    try:
        with connection:
            connection.settimeout(_remaining(deadline_ns, clock_ns))
            connection.sendall(wire)
            while True:
                connection.settimeout(_remaining(deadline_ns, clock_ns))
                chunk = connection.recv(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    return "invalid_response", "oversized_response", b""
                chunks.append(chunk)
    except TimeoutError:
        return "timeout", "receive_timeout", b""
    except OSError as exc:
        return "connection_reset", type(exc).__name__, b""
    return "", None, b"".join(chunks)


def _classify(
    data: bytes, planned: PlannedRequest, model_sha256: str
) -> tuple[str, str | None, int | None, str | None, str | None]:
    head, separator, body = data.partition(b"\r\n\r\n")
    if not separator:
        return "connection_reset", "closed_before_response", None, None, None
    status_line, *header_lines = head.split(b"\r\n")
    match = _STATUS.fullmatch(status_line)
    if match is None:
        return "invalid_response", "status_line", None, None, None
    status = int(match.group(1))
    headers: dict[str, list[str]] = {}
    for line in header_lines:
        name, colon, value = line.partition(b":")
        if not colon or not name or not line.isascii():
            return "invalid_response", "header_line", status, None, None
        headers.setdefault(name.decode().strip().lower(), []).append(
            value.decode().strip()
        )
    lengths = headers.get("content-length", [])
    if len(lengths) != 1 or _LENGTH.fullmatch(lengths[0]) is None:
        return "invalid_response", "content_length", status, None, None
    if len(body) < int(lengths[0]):
        return "connection_reset", "truncated_body", status, None, None
    if len(body) > int(lengths[0]):
        return "invalid_response", "excess_body", status, None, None
    if status != 200:
        return "http_error", f"status_{status}", status, None, None
    lifetimes = headers.get(LIFETIME_HEADER.lower(), [])
    try:
        if headers.get("content-type") != ["application/json"] or len(lifetimes) != 1:
            raise ServingRecoveryError("response headers break the protocol")
        lifetime_id = require_lifetime_id(lifetimes[0])
        payload = strict_json_loads(body, name="response")
        if type(payload) is not dict or set(payload) != _RESPONSE_FIELDS:
            raise ServingRecoveryError("response fields do not match schema")
        if (
            {key: payload[key] for key in ("prompt", "seed", "max_new_tokens")}
            != planned.request.to_mapping()
            or payload["model_sha256"] != model_sha256
            or payload["finish_reason"] not in ("eos", "length")
            or type(payload["output"]) is not str
            or canonical_json_bytes(payload) != body
        ):
            raise ServingRecoveryError("response does not answer this request")
    except ServingRecoveryError:
        return "invalid_response", "response_body", status, None, None
    return "ok", None, status, lifetime_id, sha256_hex(body)


def send_generation_attempt(
    endpoint: str,
    planned: PlannedRequest,
    *,
    attempt: int,
    deadline_ms: int,
    model_sha256: str,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> AttemptRecord:
    """Send one bounded attempt carrying the request's idempotency key."""

    _, port = loopback_address(endpoint)
    if not isinstance(planned, PlannedRequest):
        raise TypeError("planned must be PlannedRequest")
    require_int(attempt, "attempt", 1, MAX_ATTEMPT)
    require_int(deadline_ms, "deadline_ms", 1, 60_000)
    require_digest(model_sha256, "model_sha256")
    body = planned.request.body_bytes()
    wire = (
        f"POST /generate HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        f"{KEY_HEADER}: {planned.idempotency_key}\r\n{ATTEMPT_HEADER}: {attempt}\r\n"
        "\r\n"
    ).encode("ascii") + body
    started_ns = clock_ns()
    deadline_ns = started_ns + deadline_ms * 1_000_000
    outcome, error, data = _exchange(port, wire, deadline_ns, clock_ns)
    status = lifetime_id = digest = None
    if not outcome:
        outcome, error, status, lifetime_id, digest = _classify(
            data, planned, model_sha256
        )
    finished_ns = clock_ns()
    if finished_ns > deadline_ns and outcome != "timeout":
        outcome, error, status = "timeout", "deadline_elapsed", None
        lifetime_id = digest = None
    return AttemptRecord(
        request_index=planned.index,
        idempotency_key=planned.idempotency_key,
        attempt=attempt,
        port=port,
        started_ns=started_ns,
        finished_ns=finished_ns,
        outcome=outcome,
        error=error,
        http_status=status,
        lifetime_id=lifetime_id,
        response_sha256=digest,
    )
