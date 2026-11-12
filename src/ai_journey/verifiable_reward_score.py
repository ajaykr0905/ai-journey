"""Auditable composition of format and correctness rewards."""

from __future__ import annotations

from dataclasses import dataclass

from .verifiable_reward_contract import VerifiableRewardError
from .verifiable_reward_registry import VerificationOutcome


@dataclass(frozen=True)
class RewardWeights:
    format: float = 0.2
    correctness: float = 0.8

    def __post_init__(self):
        if (
            self.format < 0
            or self.correctness < 0
            or self.format + self.correctness != 1.0
        ):
            raise VerifiableRewardError(
                "reward weights must be non-negative and sum to one"
            )


def score_outcome(
    outcome: VerificationOutcome, weights: RewardWeights = RewardWeights()
) -> float:
    return (weights.format if outcome.format_valid else 0.0) + (
        weights.correctness if outcome.correct else 0.0
    )
