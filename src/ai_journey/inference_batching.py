"""Deterministic continuous-batching schedules with reserved-token admission."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .inference_kv import InferenceCapacityError, _positive_integer


@dataclass(frozen=True)
class InferenceRequest:
    """One queued request whose prompt and whole generation budget are reserved.

    Admission charges prompt plus ``max_new_tokens`` up front, so an admitted row
    cannot outgrow the resident-token budget mid-decode. Bad ids, negative
    arrivals or nonpositive token counts raise InferenceCapacityError.
    """

    request_id: str
    arrival_tick: int
    prompt_tokens: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.request_id) is None
        ):
            raise InferenceCapacityError("request_id must be a bounded identifier")
        if type(self.arrival_tick) is not int or self.arrival_tick < 0:
            raise InferenceCapacityError("arrival_tick must be a nonnegative integer")
        _positive_integer(self.prompt_tokens, "prompt_tokens")
        _positive_integer(self.max_new_tokens, "max_new_tokens")

    @property
    def reserved_tokens(self) -> int:
        """Return the prompt plus full generation tokens charged at admission."""

        return self.prompt_tokens + self.max_new_tokens


@dataclass(frozen=True)
class BatchStep:
    """One scheduler tick: rows admitted, rows decoded and rows finished.

    ``reserved_tokens`` is the reservation held while the tick decodes: counted
    after admission and before finished rows release their budget.
    """

    tick: int
    admitted: tuple[str, ...]
    decode: tuple[str, ...]
    completed: tuple[str, ...]
    reserved_tokens: int


@dataclass(frozen=True)
class RequestTiming:
    """Per-request latency in scheduler ticks, splitting queueing from service."""

    request_id: str
    arrival_tick: int
    admitted_tick: int
    completed_tick: int

    @property
    def queue_ticks(self) -> int:
        """Return ticks spent waiting for a batch slot and token budget."""

        return self.admitted_tick - self.arrival_tick

    @property
    def service_ticks(self) -> int:
        """Return ticks from admission to completion, counting both ends.

        Admitted rows decode once per tick until done, so this equals the
        request's ``max_new_tokens``.
        """

        return self.completed_tick - self.admitted_tick + 1


def chunk_prefill(
    request: InferenceRequest, *, max_chunk_tokens: int
) -> tuple[int, ...]:
    """Split one prompt into bounded prefill chunks without padding."""

    if not isinstance(request, InferenceRequest):
        raise TypeError("request must be InferenceRequest")
    _positive_integer(max_chunk_tokens, "max_chunk_tokens")
    full, remainder = divmod(request.prompt_tokens, max_chunk_tokens)
    chunks = (max_chunk_tokens,) * full
    return chunks + ((remainder,) if remainder else ())


def simulate_continuous_batching(
    requests: tuple[InferenceRequest, ...],
    *,
    max_batch_size: int,
    max_resident_tokens: int,
) -> tuple[BatchStep, ...]:
    """Admit waiting requests as slots free and decode each active row once."""

    if type(requests) is not tuple or not requests:
        raise InferenceCapacityError("requests must be a nonempty tuple")
    _positive_integer(max_batch_size, "max_batch_size")
    _positive_integer(max_resident_tokens, "max_resident_tokens")
    if len({request.request_id for request in requests}) != len(requests):
        raise InferenceCapacityError("request ids must be unique")
    if any(request.reserved_tokens > max_resident_tokens for request in requests):
        raise InferenceCapacityError("a request exceeds the resident-token budget")
    waiting = sorted(
        requests, key=lambda request: (request.arrival_tick, request.request_id)
    )
    active: dict[str, tuple[InferenceRequest, int]] = {}
    steps: list[BatchStep] = []
    tick = 0
    while waiting or active:
        admitted: list[str] = []
        reserved = sum(request.reserved_tokens for request, _ in active.values())
        for request in tuple(waiting):
            if request.arrival_tick > tick:
                break
            if len(active) >= max_batch_size:
                break
            if reserved + request.reserved_tokens > max_resident_tokens:
                continue
            waiting.remove(request)
            active[request.request_id] = (request, 0)
            reserved += request.reserved_tokens
            admitted.append(request.request_id)
        decoded = tuple(active)
        completed: list[str] = []
        for request_id, (request, generated) in tuple(active.items()):
            generated += 1
            if generated == request.max_new_tokens:
                del active[request_id]
                completed.append(request_id)
            else:
                active[request_id] = (request, generated)
        steps.append(
            BatchStep(tick, tuple(admitted), decoded, tuple(completed), reserved)
        )
        tick += 1
    return tuple(steps)


def summarize_batch_timing(
    requests: tuple[InferenceRequest, ...], steps: tuple[BatchStep, ...]
) -> tuple[RequestTiming, ...]:
    """Derive per-request queue and service ticks from a complete schedule."""

    arrivals = {request.request_id: request.arrival_tick for request in requests}
    admitted: dict[str, int] = {}
    completed: dict[str, int] = {}
    for step in steps:
        for request_id in step.admitted:
            admitted[request_id] = step.tick
        for request_id in step.completed:
            completed[request_id] = step.tick
    if set(arrivals) != set(admitted) or set(arrivals) != set(completed):
        raise InferenceCapacityError("schedule is incomplete for the declared requests")
    return tuple(
        RequestTiming(request_id, arrival, admitted[request_id], completed[request_id])
        for request_id, arrival in arrivals.items()
    )
