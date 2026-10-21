"""Bounded local inference service contracts for the distributed training lab."""

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import re

from .ddp_contracts import integer
from .ddp_generation import SamplingConfig


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
