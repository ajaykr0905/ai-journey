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

    @property
    def queue_time_ns(self) -> int:
        return self.started_ns - self.received_ns

    @property
    def service_time_ns(self) -> int:
        return self.completed_ns - self.started_ns

    @property
    def time_to_first_token_ns(self) -> int | None:
        if self.first_token_ns is None:
            return None
        return self.first_token_ns - self.received_ns


@dataclass(frozen=True)
class RequestObservation:
    """Bind a terminal outcome and bounded counters to request timing."""

    request_id: str
    outcome: ServingOutcome
    timing: RequestTiming
    prompt_tokens: int
    generated_tokens: int
    cache_hit: bool
    retry_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or self.request_id != self.request_id.strip()
        ):
            raise ServingObservationError("request_id must be canonical nonempty text")
        if not isinstance(self.outcome, ServingOutcome):
            raise TypeError("outcome must be ServingOutcome")
        if not isinstance(self.timing, RequestTiming):
            raise TypeError("timing must be RequestTiming")
        if (
            isinstance(self.prompt_tokens, bool)
            or not isinstance(self.prompt_tokens, int)
            or self.prompt_tokens <= 0
        ):
            raise ServingObservationError("prompt_tokens must be positive")
        if (
            isinstance(self.generated_tokens, bool)
            or not isinstance(self.generated_tokens, int)
            or self.generated_tokens < 0
        ):
            raise ServingObservationError("generated_tokens must be nonnegative")
        if self.outcome is ServingOutcome.SUCCESS and self.generated_tokens == 0:
            raise ServingObservationError("successful requests must generate tokens")
        if not isinstance(self.cache_hit, bool):
            raise TypeError("cache_hit must be boolean")
        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or self.retry_count < 0
        ):
            raise ServingObservationError("retry_count must be nonnegative")
