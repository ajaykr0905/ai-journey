"""Measured request lifecycle records for bounded serving experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256

from .serving_trace import RequestSpec


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


def validate_observation_for_request(
    observation: RequestObservation, request: RequestSpec
) -> RequestObservation:
    """Reject measurements that do not belong to the scheduled request."""

    if not isinstance(observation, RequestObservation):
        raise TypeError("observation must be RequestObservation")
    if not isinstance(request, RequestSpec):
        raise TypeError("request must be RequestSpec")
    if observation.request_id != request.request_id:
        raise ServingObservationError("observation request ID does not match trace")
    if observation.prompt_tokens != request.prompt_tokens:
        raise ServingObservationError("observed prompt tokens do not match trace")
    if observation.generated_tokens > request.max_output_tokens:
        raise ServingObservationError("generated tokens exceed the scheduled maximum")
    return observation


def validate_observations(
    observations: tuple[RequestObservation, ...],
) -> tuple[RequestObservation, ...]:
    """Require a nonempty observation tuple ordered by receipt time."""

    if not isinstance(observations, tuple) or not observations:
        raise TypeError("observations must be a nonempty RequestObservation tuple")
    previous_received = -1
    for observation in observations:
        if not isinstance(observation, RequestObservation):
            raise TypeError("observation entries must be RequestObservation")
        if observation.timing.received_ns < previous_received:
            raise ServingObservationError(
                "observations must be ordered by receipt time"
            )
        previous_received = observation.timing.received_ns
    return observations


def require_unique_observation_ids(
    observations: tuple[RequestObservation, ...],
) -> tuple[RequestObservation, ...]:
    """Reject duplicate terminal records for the same scheduled request."""

    validate_observations(observations)
    request_ids = [observation.request_id for observation in observations]
    if len(request_ids) != len(set(request_ids)):
        raise ServingObservationError("observation request IDs must be unique")
    return observations


def censored_observations(
    observations: tuple[RequestObservation, ...],
) -> tuple[RequestObservation, ...]:
    """Return requests lacking an ordinary successful terminal observation."""

    require_unique_observation_ids(observations)
    return tuple(
        observation
        for observation in observations
        if observation.outcome is not ServingOutcome.SUCCESS
    )


def canonical_observation_bytes(
    observations: tuple[RequestObservation, ...],
) -> bytes:
    """Serialize ordered unique observations with stable JSON rules."""

    require_unique_observation_ids(observations)
    return json.dumps(
        [asdict(observation) for observation in observations],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def serving_observation_digest(
    observations: tuple[RequestObservation, ...],
) -> str:
    """Return the stable identity of measured request lifecycle evidence."""

    return sha256(canonical_observation_bytes(observations)).hexdigest()
