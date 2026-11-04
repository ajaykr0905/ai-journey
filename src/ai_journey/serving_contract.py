"""Immutable contracts for reproducible serving measurements."""

from __future__ import annotations

from dataclasses import dataclass


class ServingContractError(ValueError):
    """Raised when a serving run contract is ambiguous or unbounded."""


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ServingContractError(f"{name} must be positive")
    return value


def _nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if not value.strip() or value != value.strip():
        raise ServingContractError(f"{name} must be nonempty canonical text")
    return value


@dataclass(frozen=True)
class EngineIdentity:
    """Pin the serving engine implementation under measurement."""

    name: str
    version: str
    revision: str

    def __post_init__(self) -> None:
        for name in ("name", "version", "revision"):
            _nonempty_text(getattr(self, name), name)


@dataclass(frozen=True)
class ModelIdentity:
    """Pin model, tokenizer, and immutable revision identifiers."""

    repository: str
    revision: str
    tokenizer_revision: str
    parameter_count: int

    def __post_init__(self) -> None:
        for name in ("repository", "revision", "tokenizer_revision"):
            _nonempty_text(getattr(self, name), name)
        _positive_integer(self.parameter_count, "parameter_count")


@dataclass(frozen=True)
class WorkloadShape:
    """Bound the request count, token demand, and client concurrency."""

    request_count: int
    prompt_tokens_per_request: int
    output_tokens_per_request: int
    max_concurrency: int

    def __post_init__(self) -> None:
        for name in (
            "request_count",
            "prompt_tokens_per_request",
            "output_tokens_per_request",
            "max_concurrency",
        ):
            _positive_integer(getattr(self, name), name)
        if self.max_concurrency > self.request_count:
            raise ServingContractError("max_concurrency cannot exceed request_count")

    @property
    def total_requested_tokens(self) -> int:
        """Return the bounded prompt-plus-output token demand."""

        return self.request_count * (
            self.prompt_tokens_per_request + self.output_tokens_per_request
        )
