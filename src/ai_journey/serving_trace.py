"""Deterministic, content-private request traces for serving experiments."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


class ServingTraceError(ValueError):
    """Raised when a request trace is ambiguous, unsafe, or unbounded."""


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ServingTraceError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class RequestSpec:
    """Describe one scheduled request without retaining raw prompt content."""

    request_id: str
    arrival_ms: int
    prompt_tokens: int
    max_output_tokens: int
    prompt_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or self.request_id != self.request_id.strip()
        ):
            raise ServingTraceError("request_id must be canonical nonempty text")
        if (
            isinstance(self.arrival_ms, bool)
            or not isinstance(self.arrival_ms, int)
            or self.arrival_ms < 0
        ):
            raise ServingTraceError("arrival_ms must be a nonnegative integer")
        _positive_integer(self.prompt_tokens, "prompt_tokens")
        _positive_integer(self.max_output_tokens, "max_output_tokens")
        if (
            not isinstance(self.prompt_sha256, str)
            or len(self.prompt_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.prompt_sha256
            )
        ):
            raise ServingTraceError("prompt_sha256 must be a lowercase SHA-256 digest")


def validate_trace(trace: tuple[RequestSpec, ...]) -> tuple[RequestSpec, ...]:
    """Require a nonempty, arrival-ordered trace with unique request IDs."""

    if not isinstance(trace, tuple) or not trace:
        raise TypeError("trace must be a nonempty RequestSpec tuple")
    request_ids: set[str] = set()
    previous_arrival = -1
    for request in trace:
        if not isinstance(request, RequestSpec):
            raise TypeError("trace entries must be RequestSpec")
        if request.request_id in request_ids:
            raise ServingTraceError("request IDs must be unique")
        if request.arrival_ms < previous_arrival:
            raise ServingTraceError("trace arrivals must be ordered")
        request_ids.add(request.request_id)
        previous_arrival = request.arrival_ms
    return trace


def constant_rate_trace(
    *,
    request_count: int,
    interval_ms: int,
    prompt_tokens: int,
    max_output_tokens: int,
    prompt_sha256: str,
) -> tuple[RequestSpec, ...]:
    """Generate an exact fixed-interval arrival trace."""

    _positive_integer(request_count, "request_count")
    _positive_integer(interval_ms, "interval_ms")
    trace = tuple(
        RequestSpec(
            request_id=f"request-{index + 1:06d}",
            arrival_ms=index * interval_ms,
            prompt_tokens=prompt_tokens,
            max_output_tokens=max_output_tokens,
            prompt_sha256=prompt_sha256,
        )
        for index in range(request_count)
    )
    return validate_trace(trace)


def burst_trace(
    *,
    burst_count: int,
    requests_per_burst: int,
    inter_burst_ms: int,
    prompt_tokens: int,
    max_output_tokens: int,
    prompt_sha256: str,
) -> tuple[RequestSpec, ...]:
    """Generate tied arrivals separated by a fixed inter-burst interval."""

    _positive_integer(burst_count, "burst_count")
    _positive_integer(requests_per_burst, "requests_per_burst")
    _positive_integer(inter_burst_ms, "inter_burst_ms")
    trace = tuple(
        RequestSpec(
            request_id=f"burst-{burst + 1:04d}-{member + 1:04d}",
            arrival_ms=burst * inter_burst_ms,
            prompt_tokens=prompt_tokens,
            max_output_tokens=max_output_tokens,
            prompt_sha256=prompt_sha256,
        )
        for burst in range(burst_count)
        for member in range(requests_per_burst)
    )
    return validate_trace(trace)


def arrival_rate_per_second(trace: tuple[RequestSpec, ...]) -> Fraction:
    """Return the exact inter-arrival rate between first and last requests."""

    validate_trace(trace)
    if len(trace) < 2 or trace[-1].arrival_ms == trace[0].arrival_ms:
        raise ServingTraceError("arrival rate requires a positive trace span")
    return Fraction((len(trace) - 1) * 1000, trace[-1].arrival_ms - trace[0].arrival_ms)


def offered_token_rate_per_second(
    trace: tuple[RequestSpec, ...], duration_ms: int
) -> Fraction:
    """Return prompt-plus-requested-output tokens offered per second."""

    validate_trace(trace)
    _positive_integer(duration_ms, "duration_ms")
    if duration_ms <= trace[-1].arrival_ms:
        raise ServingTraceError("duration_ms must extend beyond the final arrival")
    offered_tokens = sum(
        request.prompt_tokens + request.max_output_tokens for request in trace
    )
    return Fraction(offered_tokens * 1000, duration_ms)


def split_warmup(
    trace: tuple[RequestSpec, ...], warmup_requests: int
) -> tuple[tuple[RequestSpec, ...], tuple[RequestSpec, ...]]:
    """Split an ordered trace without allowing warmup to consume all samples."""

    validate_trace(trace)
    _positive_integer(warmup_requests, "warmup_requests")
    if warmup_requests >= len(trace):
        raise ServingTraceError("warmup must leave at least one measured request")
    return trace[:warmup_requests], trace[warmup_requests:]


def slice_trace_window(
    trace: tuple[RequestSpec, ...], *, start_ms: int, end_ms: int
) -> tuple[RequestSpec, ...]:
    """Select requests in one half-open absolute arrival window."""

    validate_trace(trace)
    if (
        isinstance(start_ms, bool)
        or not isinstance(start_ms, int)
        or start_ms < 0
        or isinstance(end_ms, bool)
        or not isinstance(end_ms, int)
        or end_ms <= start_ms
    ):
        raise ServingTraceError("trace window must be nonnegative and increasing")
    selected = tuple(
        request for request in trace if start_ms <= request.arrival_ms < end_ms
    )
    if not selected:
        raise ServingTraceError("trace window contains no requests")
    return selected
