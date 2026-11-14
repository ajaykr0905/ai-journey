"""Hash-chained lifecycle trace for serving failure-injection experiments.

Each event binds the canonical digest of the whole preceding trace, as in the
worker recovery trace. Verification replays a lifecycle state machine: the
first lifetime is spawned, readiness follows its spawn, the single fault hits a
ready lifetime whose observed exit is that SIGKILL, an exit is observed before
any restart, a restart names its exited parent and the next lifetime number,
recovery is declared only on a ready post-fault lifetime, and ``finished`` or
``exhausted`` is terminal. Payloads have one exact schema per event kind.
"""

from __future__ import annotations

import signal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .serving_recovery_config import (
    ServingRecoveryError,
    canonical_sha256,
    require_digest,
    require_int,
)
from .serving_recovery_identity import (
    lifetime_number,
    require_lifetime_id,
    require_run_id,
)

_PAYLOADS: dict[str, dict[str, type]] = {
    "spawn": {"pid": int},
    "ready": {"pid": int, "port": int},
    "unresponsive": {"pid": int},
    "fault": {"pid": int, "signal": str, "acknowledged_responses": int},
    "exit_observed": {"pid": int, "exit_code": int},
    "restart": {"pid": int, "parent_lifetime_id": str, "backoff_ms": int},
    "exhausted": {"restarts": int},
    "recovered": {"consecutive_successes": int, "slo_latency_ms": int},
    "finished": {"pid": int, "exit_code": int},
}
TRACE_KINDS = tuple(_PAYLOADS)
_RECORD_FIELDS = frozenset(
    {
        "sequence",
        "run_id",
        "lifetime_id",
        "kind",
        "monotonic_ns",
        "payload",
        "previous_sha256",
        "event_sha256",
    }
)


@dataclass(frozen=True)
class ServingLifecycleEvent:
    """One ordered lifecycle observation about one server lifetime."""

    sequence: int
    run_id: str
    lifetime_id: str
    kind: str
    monotonic_ns: int
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_int(self.sequence, "sequence", 0, 4095)
        require_run_id(self.run_id)
        require_lifetime_id(self.lifetime_id)
        require_int(self.monotonic_ns, "monotonic_ns", 0, 2**63 - 1)
        if self.kind not in _PAYLOADS:
            raise ServingRecoveryError("trace event kind is unsupported")
        schema = _PAYLOADS[self.kind]
        if not isinstance(self.payload, Mapping) or set(self.payload) != set(schema):
            raise ServingRecoveryError(f"{self.kind} payload does not match schema")
        for name, expected in schema.items():
            if type(self.payload[name]) is not expected:
                raise ServingRecoveryError(f"{self.kind}.{name} has the wrong type")
        object.__setattr__(self, "payload", dict(self.payload))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "run_id": self.run_id,
            "lifetime_id": self.lifetime_id,
            "kind": self.kind,
            "monotonic_ns": self.monotonic_ns,
            "payload": dict(self.payload),
        }


def chain_serving_event(
    event: ServingLifecycleEvent, previous_sha256: str | None
) -> dict[str, Any]:
    """Bind one event to the digest of the accepted trace prefix."""

    if not isinstance(event, ServingLifecycleEvent):
        raise TypeError("event must be ServingLifecycleEvent")
    if previous_sha256 is not None:
        require_digest(previous_sha256, "previous_sha256")
    record = {**event.to_mapping(), "previous_sha256": previous_sha256}
    record["event_sha256"] = canonical_sha256(record)
    return record


def chain_serving_trace(
    events: Sequence[ServingLifecycleEvent],
) -> list[dict[str, Any]]:
    """Chain a whole event sequence and verify the resulting lifecycle."""

    records: list[dict[str, Any]] = []
    for event in events:
        records.append(
            chain_serving_event(event, records[-1]["event_sha256"] if records else None)
        )
    verify_serving_trace(records, require_finished=False)
    return records


class _Lifecycle:
    """Replay lifecycle transitions and reject impossible orderings."""

    def __init__(self) -> None:
        self.lifetime: str | None = None
        self.pid: int | None = None
        self.state = "none"
        self.fault_lifetime: str | None = None
        self.recovered = False
        self.terminal = False

    def apply(self, event: ServingLifecycleEvent) -> None:
        if self.terminal:
            raise ServingRecoveryError("trace continues after a terminal event")
        payload = event.payload
        if self.lifetime is None and event.kind != "spawn":
            raise ServingRecoveryError("trace must begin by spawning lifetime 1")
        if event.kind in ("spawn", "restart"):
            self._begin(event)
            return
        if event.lifetime_id != self.lifetime:
            raise ServingRecoveryError(f"{event.kind} names another lifetime")
        if "pid" in payload and payload["pid"] != self.pid:
            raise ServingRecoveryError(f"{event.kind} names another process")
        allowed = {
            "ready": {"spawned"},
            "unresponsive": {"ready"},
            "fault": {"ready"},
            "exit_observed": {"spawned", "ready", "unresponsive", "faulted"},
            "exhausted": {"exited"},
            "recovered": {"ready"},
            "finished": {"ready"},
        }[event.kind]
        if self.state not in allowed:
            raise ServingRecoveryError(f"{event.kind} cannot follow {self.state}")
        if event.kind == "ready":
            require_int(payload["port"], "port", 1, 65535)
            self.state = "ready"
        elif event.kind == "unresponsive":
            self.state = "unresponsive"
        elif event.kind == "fault":
            if self.fault_lifetime is not None:
                raise ServingRecoveryError("trace contains more than one fault")
            if payload["signal"] != "SIGKILL":
                raise ServingRecoveryError("fault signal must be SIGKILL")
            require_int(payload["acknowledged_responses"], "acknowledged", 1, 255)
            self.fault_lifetime, self.state = event.lifetime_id, "faulted"
        elif event.kind == "exit_observed":
            if self.state == "faulted" and payload["exit_code"] != -signal.SIGKILL:
                raise ServingRecoveryError(
                    "the faulted lifetime did not exit by SIGKILL"
                )
            self.state = "exited"
        elif event.kind == "recovered":
            if self.fault_lifetime is None or self.recovered:
                raise ServingRecoveryError("recovery must follow exactly one fault")
            if lifetime_number(self.lifetime) <= lifetime_number(self.fault_lifetime):
                raise ServingRecoveryError("recovery needs a post-fault lifetime")
            require_int(payload["consecutive_successes"], "successes", 1, 64)
            require_int(payload["slo_latency_ms"], "slo_latency_ms", 1, 10_000)
            self.recovered = True
        else:
            if event.kind == "exhausted":
                require_int(payload["restarts"], "restarts", 1, 5)
            self.terminal = True

    def _begin(self, event: ServingLifecycleEvent) -> None:
        number = lifetime_number(event.lifetime_id)
        if event.kind == "spawn":
            if self.lifetime is not None or number != 1:
                raise ServingRecoveryError("only the first lifetime is spawned")
        else:
            if self.state != "exited":
                raise ServingRecoveryError("restart must follow an observed exit")
            if event.payload["parent_lifetime_id"] != self.lifetime:
                raise ServingRecoveryError("restart names another parent lifetime")
            if number != lifetime_number(self.lifetime) + 1:
                raise ServingRecoveryError("restart skips a lifetime number")
            require_int(event.payload["backoff_ms"], "backoff_ms", 0, 80_000)
        require_int(event.payload["pid"], "pid", 1, 2**31 - 1)
        self.lifetime, self.pid, self.state = (
            event.lifetime_id,
            event.payload["pid"],
            "spawned",
        )


def verify_serving_trace(
    records: Sequence[Mapping[str, Any]], *, require_finished: bool = True
) -> tuple[ServingLifecycleEvent, ...]:
    """Verify exact records, digest ancestry, time order and lifecycle order."""

    if type(require_finished) is not bool:
        raise TypeError("require_finished must be a boolean")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence")
    if not records:
        raise ServingRecoveryError("serving trace must not be empty")
    events = []
    lifecycle = _Lifecycle()
    previous = None
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
            raise ServingRecoveryError("trace record fields do not match schema")
        event = ServingLifecycleEvent(
            **{
                key: value
                for key, value in record.items()
                if key not in ("previous_sha256", "event_sha256")
            }
        )
        if event.sequence != index:
            raise ServingRecoveryError("trace sequence is not contiguous")
        if record["previous_sha256"] != previous:
            raise ServingRecoveryError("trace ancestry changed")
        if dict(record) != chain_serving_event(event, previous):
            raise ServingRecoveryError("trace digest does not match its event")
        if events and event.run_id != events[0].run_id:
            raise ServingRecoveryError("trace run identity changed")
        if events and event.monotonic_ns < events[-1].monotonic_ns:
            raise ServingRecoveryError("trace time moved backwards")
        lifecycle.apply(event)
        previous = record["event_sha256"]
        events.append(event)
    if require_finished and (
        events[-1].kind != "finished" or lifecycle.fault_lifetime is None
    ):
        raise ServingRecoveryError("a complete trace has one fault and finishes")
    return tuple(events)
