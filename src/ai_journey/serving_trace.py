"""Deterministic, content-private request traces for serving experiments."""

from __future__ import annotations

from dataclasses import dataclass


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
