"""Deterministic fault schedules and recovery observations for serving labs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ServingFaultError(ValueError):
    """Raised when fault or recovery evidence is ambiguous."""


class FaultKind(str, Enum):
    """Fixed fault classes supported by the serving experiment protocol."""

    PROCESS_KILL = "process_kill"
    ENGINE_RESTART = "engine_restart"
    NETWORK_PAUSE = "network_pause"
    DEVICE_OOM = "device_oom"


@dataclass(frozen=True)
class FaultEvent:
    """Schedule one bounded fault against one explicit target."""

    fault_id: str
    kind: FaultKind
    target: str
    start_ns: int
    duration_ns: int

    def __post_init__(self) -> None:
        for value, name in ((self.fault_id, "fault_id"), (self.target, "target")):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ServingFaultError(f"{name} must be canonical nonempty text")
        if not isinstance(self.kind, FaultKind):
            raise TypeError("kind must be FaultKind")
        if (
            isinstance(self.start_ns, bool)
            or not isinstance(self.start_ns, int)
            or self.start_ns < 0
        ):
            raise ServingFaultError("start_ns must be nonnegative")
        if (
            isinstance(self.duration_ns, bool)
            or not isinstance(self.duration_ns, int)
            or self.duration_ns <= 0
        ):
            raise ServingFaultError("duration_ns must be positive")

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.duration_ns
