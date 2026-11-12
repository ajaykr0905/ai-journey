"""Bounded controls for deterministic verifiable-reward experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class VerifiableRewardError(ValueError):
    """Raised when reward evidence violates its declared contract."""


def _require_int(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerifiableRewardError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise VerifiableRewardError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class RewardRunConfig:
    """Resource and determinism limits for a fixture-backed reward run."""

    seed: int
    candidates_per_task: int
    max_tasks: int
    max_response_bytes: int
    max_attempts: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RewardRunConfig":
        expected = {
            "seed",
            "candidates_per_task",
            "max_tasks",
            "max_response_bytes",
            "max_attempts",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise VerifiableRewardError(
                f"config fields mismatch: missing={missing}, extra={extra}"
            )
        return cls(
            seed=_require_int(value["seed"], "seed", minimum=0, maximum=2**32 - 1),
            candidates_per_task=_require_int(
                value["candidates_per_task"],
                "candidates_per_task",
                minimum=2,
                maximum=32,
            ),
            max_tasks=_require_int(
                value["max_tasks"], "max_tasks", minimum=1, maximum=1_000
            ),
            max_response_bytes=_require_int(
                value["max_response_bytes"],
                "max_response_bytes",
                minimum=16,
                maximum=1_000_000,
            ),
            max_attempts=_require_int(
                value["max_attempts"], "max_attempts", minimum=1, maximum=8
            ),
        )
