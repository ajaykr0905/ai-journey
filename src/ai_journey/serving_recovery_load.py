"""Bounded concurrent load whose failed attempts retry under one idempotency key.

At most ``concurrency`` logical requests are in flight. A request retries every
failed attempt after a capped exponential backoff until it succeeds or spends
its retry budget. Each attempt carries the request's idempotency key and reads
the currently published endpoint, which stays stale while a replacement server
starts. The ledger appends every attempt and calls its acknowledgement hook
while holding its reentrant lock, so a fault hook sees an exact acknowledged
count and may read the ledger, but no other thread records meanwhile.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence

from .serving_recovery_client import (
    AttemptRecord,
    PlannedRequest,
    send_generation_attempt,
)
from .serving_recovery_config import (
    ServingRecoveryConfig,
    ServingRecoveryError,
    require_digest,
)


class AttemptLedger:
    """Thread-safe append-only attempt log with an acknowledgement counter."""

    def __init__(
        self, *, on_acknowledged: Callable[[int], object] | None = None
    ) -> None:
        if on_acknowledged is not None and not callable(on_acknowledged):
            raise TypeError("on_acknowledged must be callable")
        self._on_acknowledged = on_acknowledged
        self._lock = threading.RLock()
        self._attempts: list[AttemptRecord] = []
        self._acknowledged = 0

    def record(self, attempt: AttemptRecord) -> int:
        """Append one attempt; return the acknowledged count after it."""

        if not isinstance(attempt, AttemptRecord):
            raise TypeError("attempt must be AttemptRecord")
        with self._lock:
            self._attempts.append(attempt)
            if attempt.outcome == "ok":
                self._acknowledged += 1
                if self._on_acknowledged is not None:
                    self._on_acknowledged(self._acknowledged)
            return self._acknowledged

    @property
    def acknowledged(self) -> int:
        with self._lock:
            return self._acknowledged

    def attempts(self) -> tuple[AttemptRecord, ...]:
        with self._lock:
            return tuple(self._attempts)


def run_with_retries(
    planned: PlannedRequest,
    *,
    endpoint: Callable[[], str],
    config: ServingRecoveryConfig,
    model_sha256: str,
    ledger: AttemptLedger,
    deadline_ns: int,
    stop: threading.Event | None = None,
    wait: Callable[[float], bool] | None = None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    send: Callable[..., AttemptRecord] = send_generation_attempt,
) -> AttemptRecord:
    """Attempt one request until success or retry-budget exhaustion."""

    if not isinstance(planned, PlannedRequest):
        raise TypeError("planned must be PlannedRequest")
    if not isinstance(config, ServingRecoveryConfig):
        raise TypeError("config must be ServingRecoveryConfig")
    require_digest(model_sha256, "model_sha256")
    stop = stop if stop is not None else threading.Event()
    wait = wait if wait is not None else stop.wait
    attempt = 0
    while True:
        attempt += 1
        if stop.is_set() or clock_ns() >= deadline_ns:
            raise ServingRecoveryError("load stopped before a request finished")
        record = send(
            endpoint(),
            planned,
            attempt=attempt,
            deadline_ms=config.request_deadline_ms,
            model_sha256=model_sha256,
            clock_ns=clock_ns,
        )
        if (record.idempotency_key, record.attempt) != (
            planned.idempotency_key,
            attempt,
        ):
            raise ServingRecoveryError("sender returned another attempt's record")
        ledger.record(record)
        if record.outcome == "ok" or attempt > config.retry_budget:
            return record
        if wait(config.retry_delay_ms(attempt) / 1000):
            raise ServingRecoveryError("load stopped during retry backoff")


def run_load(
    plan: Sequence[PlannedRequest],
    *,
    endpoint: Callable[[], str],
    config: ServingRecoveryConfig,
    model_sha256: str,
    ledger: AttemptLedger,
    deadline_ns: int,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    send: Callable[..., AttemptRecord] = send_generation_attempt,
) -> tuple[AttemptRecord, ...]:
    """Run every planned request with bounded concurrency and wall-clock time."""

    if [planned.index for planned in plan] != list(range(config.requests)):
        raise ServingRecoveryError("plan must contain every request index in order")
    if clock_ns() >= deadline_ns:
        raise ServingRecoveryError("load deadline has already passed")
    stop = threading.Event()
    pending = iter(plan)
    pending_lock = threading.Lock()
    errors: list[BaseException] = []

    def worker() -> None:
        while not stop.is_set():
            with pending_lock:
                planned = next(pending, None)
            if planned is None:
                return
            try:
                run_with_retries(
                    planned,
                    endpoint=endpoint,
                    config=config,
                    model_sha256=model_sha256,
                    ledger=ledger,
                    deadline_ns=deadline_ns,
                    stop=stop,
                    clock_ns=clock_ns,
                    send=send,
                )
            except BaseException as exc:  # surfaced to the caller below
                errors.append(exc)
                stop.set()
                return

    workers = [
        threading.Thread(target=worker, daemon=True) for _ in range(config.concurrency)
    ]
    for thread in workers:
        thread.start()
    grace_s = config.request_deadline_ms / 1000 + 1
    for thread in workers:
        thread.join(timeout=max(0.0, (deadline_ns - clock_ns()) / 1e9) + grace_s)
        if thread.is_alive():
            stop.set()
            for other in workers:
                other.join(timeout=grace_s)
            raise ServingRecoveryError("load exceeded its wall-clock bound")
    if errors:
        raise errors[0]
    return ledger.attempts()
