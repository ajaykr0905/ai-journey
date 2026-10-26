"""Deterministic request schedules with explicit local HTTP load budgets."""

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import math
import re

from .http_measurements import validated_request


@dataclass(frozen=True)
class HTTPWorkloadProfile:
    prompts: tuple[tuple[int, ...], ...] = ((1, 2),)
    measured_requests: int = 8
    warmup_requests: int = 1
    concurrency: int = 2
    max_new_tokens: int = 4
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int = 48
    timeout_seconds: float = 5.0
    streaming: bool = True
    cancel_every: int = 0
    cancel_after_tokens: int = 1

    def __post_init__(self):
        for name, minimum, maximum in (
            ("measured_requests", 1, 128),
            ("warmup_requests", 0, 16),
            ("concurrency", 1, 8),
            ("max_new_tokens", 1, 32),
            ("seed", 0, 2**64 - 1),
            ("cancel_every", 0, 128),
            ("cancel_after_tokens", 1, 31),
        ):
            if (
                type(getattr(self, name)) is not int
                or not minimum <= getattr(self, name) <= maximum
            ):
                raise ValueError(f"invalid HTTP profile {name}")
        if (
            type(self.prompts) is not tuple
            or not 1 <= len(self.prompts) <= 16
            or any(
                type(prompt) is not tuple or not 1 <= len(prompt) <= 64
                for prompt in self.prompts
            )
        ):
            raise ValueError(
                "HTTP profile requires one to sixteen bounded token prompts"
            )
        if (
            type(self.streaming) is not bool
            or self.concurrency > self.measured_requests
        ):
            raise ValueError("HTTP profile concurrency or streaming control is invalid")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 10
        ):
            raise ValueError(
                "HTTP profile timeout_seconds must be finite and in (0, 10]"
            )
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        if self.cancel_every and (
            not self.streaming
            or self.cancel_after_tokens >= self.max_new_tokens
            or self.cancel_every > self.measured_requests
        ):
            raise ValueError(
                "HTTP profile cancellation must affect a bounded streaming request"
            )
        for prompt in self.prompts:
            validated_request(
                {
                    "request_id": "profile",
                    "prompt": list(prompt),
                    "sampling": self.sampling(0),
                }
            )
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "top_p", float(self.top_p))
        tokens = (self.measured_requests + self.warmup_requests) * (
            max(map(len, self.prompts)) + self.max_new_tokens
        )
        if tokens > 8192:
            raise ValueError("HTTP profile exceeds its 8192-token request budget")
        if (
            self.warmup_requests + math.ceil(self.measured_requests / self.concurrency)
        ) * self.timeout_seconds > 120:
            raise ValueError("HTTP profile exceeds its declared 120-second I/O budget")

    def sampling(self, index):
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": (self.seed + index) % 2**64,
        }

    def to_mapping(self):
        return asdict(self) | {"prompts": [list(prompt) for prompt in self.prompts]}

    def fingerprint(self):
        return sha256(
            json.dumps(
                self.to_mapping(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()


def http_profile_from_mapping(payload):
    if (
        type(payload) is not dict
        or set(payload) != {field.name for field in fields(HTTPWorkloadProfile)}
        or type(payload["prompts"]) is not list
        or any(type(prompt) is not list for prompt in payload["prompts"])
    ):
        raise ValueError("HTTP profile fields must match the complete schema")
    return HTTPWorkloadProfile(
        **(payload | {"prompts": tuple(tuple(prompt) for prompt in payload["prompts"])})
    )


def http_request_schedule(
    profile: HTTPWorkloadProfile, run_id: str, *, phase="measured"
):
    if type(run_id) is not str or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("run_id must be a fresh UUID hex value")
    if phase not in {"warmup", "measured"}:
        raise ValueError("unsupported HTTP workload phase")
    count = profile.warmup_requests if phase == "warmup" else profile.measured_requests
    return [
        {
            "index": index,
            "phase": phase,
            "request": {
                "request_id": f"load-{run_id}-{phase[0]}{index}",
                "prompt": list(profile.prompts[index % len(profile.prompts)]),
                "sampling": profile.sampling(index),
            },
            "cancel_after_tokens": profile.cancel_after_tokens
            if phase == "measured"
            and profile.cancel_every
            and (index + 1) % profile.cancel_every == 0
            else None,
        }
        for index in range(count)
    ]
