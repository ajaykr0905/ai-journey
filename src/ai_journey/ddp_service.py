"""Bounded local inference service contracts for the distributed training lab."""

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import re
import copy
import threading
from collections import OrderedDict
import time
import math

from .ddp_contracts import integer
from .ddp_generation import SamplingConfig, generate_tokens
from .gpt2_training import gpt2_model_fingerprint


class AdmissionRejected(RuntimeError):
    def __init__(self, retry_after: float):
        super().__init__("generation token budget exhausted")
        self.retry_after = retry_after


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
        self.clock = clock
        self.updated = clock()
        self.lock = threading.Lock()

    def consume(self, cost: int):
        integer(cost, "cost", 1)
        if cost > self.capacity:
            raise ValueError("request cannot fit the admission capacity")
        with self.lock:
            now = self.clock()
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
        return cls(
            payload["request_id"],
            tuple(payload["prompt"]),
            SamplingConfig(**payload["sampling"]),
        )


class GenerationService:
    """Serve a frozen checkpoint snapshot under one serialized inference lock."""

    def __init__(self, model, *, cache_capacity: int = 128, admission=None):
        integer(cache_capacity, "cache_capacity", 1)
        if cache_capacity > 4096:
            raise ValueError("response cache exceeds lab budget")
        self._model = copy.deepcopy(model).eval()
        self._model.requires_grad_(False)
        self.model_sha256 = gpt2_model_fingerprint(self._model)
        self._lock = threading.RLock()
        self.completed_requests = 0
        self.generated_tokens = 0
        self._responses = OrderedDict()
        self._cache_capacity = cache_capacity
        self._admission = admission if admission is not None else TokenBucket()

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled=None,
        deadline=None,
        on_token=None,
    ) -> dict:
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be GenerationRequest")
        with self._lock:
            fingerprint = request.fingerprint()
            if request.request_id in self._responses:
                previous_fingerprint, previous_output = self._responses[
                    request.request_id
                ]
                if previous_fingerprint != fingerprint:
                    raise ValueError("request_id was already used with different input")
                self._responses.move_to_end(request.request_id)
                if on_token is not None:
                    for index, token in enumerate(previous_output["tokens"]):
                        on_token(index, token)
                return copy.deepcopy(previous_output)
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
        with self._lock:
            return {
                "ready": True,
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
