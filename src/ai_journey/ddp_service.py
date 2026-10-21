"""Bounded local inference service contracts for the distributed training lab."""

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import re
import copy
import threading

from .ddp_contracts import integer
from .ddp_generation import SamplingConfig, generate_tokens
from .gpt2_training import gpt2_model_fingerprint


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

    def __init__(self, model):
        self._model = copy.deepcopy(model).eval()
        self._model.requires_grad_(False)
        self.model_sha256 = gpt2_model_fingerprint(self._model)
        self._lock = threading.RLock()
        self.completed_requests = 0
        self.generated_tokens = 0

    def generate(self, request: GenerationRequest) -> dict:
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be GenerationRequest")
        with self._lock:
            output = generate_tokens(
                self._model, request.prompt, request.sampling, use_cache=True
            )
            self.completed_requests += 1
            self.generated_tokens += len(output["tokens"])
            return {
                "request_id": request.request_id,
                "model_sha256": self.model_sha256,
                **output,
            }

    def health(self) -> dict:
        with self._lock:
            return {
                "ready": True,
                "device": "cpu",
                "model_sha256": self.model_sha256,
                "completed_requests": self.completed_requests,
                "generated_tokens": self.generated_tokens,
            }
