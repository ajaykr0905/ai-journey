"""Measured request lifecycle records for bounded serving experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ServingObservationError(ValueError):
    """Raised when measured request evidence is ambiguous or inconsistent."""


class ServingOutcome(str, Enum):
    """Fixed-cardinality terminal outcomes for serving requests."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    OVERLOADED = "overloaded"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True)
class RequestTiming:
    """Record one request lifecycle on a shared monotonic nanosecond clock."""

    received_ns: int
    started_ns: int
    first_token_ns: int | None
    completed_ns: int

    def __post_init__(self) -> None:
        values = (self.received_ns, self.started_ns, self.completed_ns)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ServingObservationError(
                "request timestamps must be nonnegative integers"
            )
        if not self.received_ns <= self.started_ns <= self.completed_ns:
            raise ServingObservationError("request timestamps must be monotonic")
        if self.first_token_ns is not None and (
            isinstance(self.first_token_ns, bool)
            or not isinstance(self.first_token_ns, int)
            or not self.started_ns <= self.first_token_ns <= self.completed_ns
        ):
            raise ServingObservationError(
                "first-token timestamp must be within service"
            )
