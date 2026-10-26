"""Bounded local inference service contracts for the distributed training lab."""

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import re
import copy
import threading
from collections import OrderedDict
from contextlib import contextmanager
import time
import math

from .ddp_contracts import integer
from .ddp_generation import SamplingConfig, generate_tokens
from .gpt2_training import gpt2_model_fingerprint
from .training_models import base_model_config
from .serving_metrics import ServingMetrics


class AdmissionRejected(RuntimeError):
    def __init__(self, retry_after: float):
        super().__init__("generation token budget exhausted")
        self.retry_after = retry_after


class ServiceDraining(RuntimeError):
    """New requests cannot enter a service that is draining admitted work."""


class ServiceOverloaded(RuntimeError):
    """The bounded running-plus-waiting inference capacity has been reached."""


class TokenBucket:
    def __init__(
        self,
        capacity: int = 8192,
        refill_per_second: float = 4096,
        *,
        clock=time.monotonic,
    ):
        integer(capacity, "capacity", 1)
        if (
            isinstance(refill_per_second, bool)
            or not math.isfinite(refill_per_second)
            or refill_per_second <= 0
        ):
            raise ValueError("refill rate must be positive and finite")
        self.capacity = capacity
        self.rate = refill_per_second
        self.tokens = float(capacity)
        if not callable(clock):
            raise TypeError("admission clock must be callable")
        self.clock = clock
        self.updated = self._read_clock()
        self.lock = threading.Lock()

    def _read_clock(self):
        now = self.clock()
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(now)
        ):
            raise ValueError("admission clock must return a finite number")
        return now

    def consume(self, cost: int):
        integer(cost, "cost", 1)
        if cost > self.capacity:
            raise ValueError("request cannot fit the admission capacity")
        with self.lock:
            now = self._read_clock()
            if now < self.updated:
                raise RuntimeError("admission clock moved backwards")
            self.tokens = min(
                self.capacity, self.tokens + (now - self.updated) * self.rate
            )
            self.updated = now
            if cost > self.tokens:
                raise AdmissionRejected((cost - self.tokens) / self.rate)
            self.tokens -= cost


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt: tuple[int, ...]
    sampling: SamplingConfig

    def __post_init__(self):
        if (
            not isinstance(self.request_id, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.request_id) is None
        ):
            raise ValueError(
                "request_id must contain 1 to 64 safe identifier characters"
            )
        if not isinstance(self.prompt, tuple) or not 1 <= len(self.prompt) <= 512:
            raise ValueError("service prompt requires 1 to 512 token ids")
        for token in self.prompt:
            integer(token, "prompt token")
        if (
            not isinstance(self.sampling, SamplingConfig)
            or self.sampling.max_new_tokens > 128
        ):
            raise ValueError("service generation budget is 128 tokens")

    def fingerprint(self):
        return sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_mapping(cls, payload):
        if (
            not isinstance(payload, dict)
            or set(payload) != {"request_id", "prompt", "sampling"}
            or not isinstance(payload["sampling"], dict)
            or not isinstance(payload["prompt"], list)
        ):
            raise ValueError("request must contain request_id, prompt and sampling")
        if set(payload["sampling"]) - set(SamplingConfig.__dataclass_fields__):
            raise ValueError("unknown sampling field")
        try:
            sampling = SamplingConfig(**payload["sampling"])
        except OverflowError as error:
            # JSON integers can fit the byte budget without being representable
            # by the floating-point sampler. Reject before service admission.
            raise ValueError(
                "sampling number exceeds finite floating-point range"
            ) from error
        return cls(
            payload["request_id"],
            tuple(payload["prompt"]),
            sampling,
        )


class GenerationService:
    """Serve a frozen checkpoint snapshot under one serialized inference lock."""

    def __init__(
        self,
        model,
        *,
        cache_capacity: int = 128,
        admission=None,
        max_inflight_requests=8,
    ):
        integer(cache_capacity, "cache_capacity", 1)
        if cache_capacity > 4096:
            raise ValueError("response cache exceeds lab budget")
        integer(max_inflight_requests, "max_inflight_requests", 1)
        if max_inflight_requests > 64:
            raise ValueError("inference admission exceeds the lab limit of 64")
        self._model = copy.deepcopy(model).eval()
        self._model.requires_grad_(False)
        self.model_sha256 = gpt2_model_fingerprint(self._model)
        self._lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._lifecycle = threading.Condition(self._stats_lock)
        self._active_requests = 0
        self._max_inflight_requests = max_inflight_requests
        self._draining = False
        self.completed_requests = 0
        self.generated_tokens = 0
        self._responses = OrderedDict()
        self._cache_capacity = cache_capacity
        self._admission = admission if admission is not None else TokenBucket()
        self.metrics = ServingMetrics()

    @contextmanager
    def _inference_slot(self, *, cancelled, deadline):
        """Honor queued controls without attempting to preempt a running model."""
        while True:
            interrupted = None
            if cancelled is not None and cancelled():
                interrupted = "cancelled"
            elif deadline is not None and time.monotonic() >= deadline:
                interrupted = "deadline"
            if interrupted is not None:
                yield interrupted
                return
            timeout = 0.05
            if deadline is not None:
                timeout = min(timeout, max(0.0, deadline - time.monotonic()))
            if self._lock.acquire(timeout=timeout):
                break
        try:
            yield None
        finally:
            self._lock.release()

    @contextmanager
    def _admitted_request(self):
        with self._lifecycle:
            if self._draining:
                raise ServiceDraining("service is draining")
            if self._active_requests >= self._max_inflight_requests:
                raise ServiceOverloaded("inference queue is full")
            self._active_requests += 1
        try:
            yield
        finally:
            with self._lifecycle:
                self._active_requests -= 1
                self._lifecycle.notify_all()

    def drain(self, *, timeout_seconds=10):
        """Stop accepting work and wait at most the declared time for admission to empty."""
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 <= timeout_seconds <= 60
        ):
            raise ValueError("drain timeout must be finite and in [0, 60]")
        with self._lifecycle:
            self._draining = True
            return self._lifecycle.wait_for(
                lambda: self._active_requests == 0, timeout=timeout_seconds
            )

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled=None,
        deadline=None,
        on_token=None,
    ) -> dict:
        started = time.perf_counter()
        observation = {"tokens": 0, "cached": False}
        outcome = "error"

        def observe(index, token):
            observation["tokens"] += 1
            if on_token is not None:
                on_token(index, token)

        try:
            if on_token is not None and not callable(on_token):
                raise TypeError("on_token must be callable")
            output = self._generate(
                request,
                cancelled=cancelled,
                deadline=deadline,
                on_token=observe,
                observation=observation,
            )
            outcome = {
                "cancelled": "cancelled",
                "deadline": "deadline",
            }.get(output["finish_reason"], "success")
            return output
        except (
            AdmissionRejected,
            ServiceDraining,
            ServiceOverloaded,
            ValueError,
            TypeError,
        ):
            outcome = "rejected"
            raise
        finally:
            self.metrics.record(
                outcome=outcome,
                duration_seconds=time.perf_counter() - started,
                generated_tokens=observation["tokens"],
                cached=observation["cached"],
            )

    def _generate(
        self,
        request: GenerationRequest,
        *,
        cancelled=None,
        deadline=None,
        on_token=None,
        observation=None,
    ) -> dict:
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be GenerationRequest")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be a callable")
        if on_token is not None and not callable(on_token):
            raise TypeError("on_token must be callable")
        if deadline is not None and (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
        ):
            raise ValueError("deadline must be a finite monotonic timestamp")
        if any(
            token >= base_model_config(self._model).vocab_size
            for token in request.prompt
        ):
            raise ValueError("prompt token exceeds vocabulary")
        with self._admitted_request(), self._inference_slot(
            cancelled=cancelled, deadline=deadline
        ) as interrupted:
            if interrupted is not None:
                return {
                    "request_id": request.request_id,
                    "model_sha256": self.model_sha256,
                    "tokens": [],
                    "prompt_tokens": len(request.prompt),
                    "finish_reason": interrupted,
                }
            fingerprint = request.fingerprint()
            if request.request_id in self._responses:
                previous_fingerprint, previous_output = self._responses[
                    request.request_id
                ]
                if previous_fingerprint != fingerprint:
                    raise ValueError("request_id was already used with different input")
                self._responses.move_to_end(request.request_id)
                observation["cached"] = True
                response = copy.deepcopy(previous_output)
                response["tokens"] = []
                for index, token in enumerate(previous_output["tokens"]):
                    if cancelled is not None and cancelled():
                        response["finish_reason"] = "cancelled"
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        response["finish_reason"] = "deadline"
                        break
                    response["tokens"].append(token)
                    if on_token is not None:
                        on_token(index, token)
                return response
            self._admission.consume(
                len(request.prompt) + request.sampling.max_new_tokens
            )
            output = generate_tokens(
                self._model,
                request.prompt,
                request.sampling,
                use_cache=True,
                cancelled=cancelled,
                deadline=deadline,
                on_token=on_token,
            )
            with self._stats_lock:
                self.completed_requests += 1
                self.generated_tokens += len(output["tokens"])
            response = {
                "request_id": request.request_id,
                "model_sha256": self.model_sha256,
                **output,
            }
            if output["finish_reason"] in ("length", "eos"):
                self._responses[request.request_id] = (
                    fingerprint,
                    copy.deepcopy(response),
                )
            while len(self._responses) > self._cache_capacity:
                self._responses.popitem(last=False)
            return response

    def health(self) -> dict:
        with self._stats_lock:
            return {
                "ready": not self._draining,
                "active_requests": self._active_requests,
                "device": "cpu",
                "model_sha256": self.model_sha256,
                "completed_requests": self.completed_requests,
                "generated_tokens": self.generated_tokens,
            }


def event_frame(event: str, payload: dict) -> bytes:
    """Encode one SSE event without permitting event-name or data-line injection."""
    if event not in ("token", "done", "error"):
        raise ValueError("unsupported generation stream event")
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return f"event: {event}\ndata: {data}\n\n".encode("ascii")
