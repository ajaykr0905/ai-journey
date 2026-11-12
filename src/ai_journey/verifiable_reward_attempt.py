"""Immutable verifier attempt records."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .verifiable_reward_contract import VerifiableRewardError

_STATUSES = frozenset(
    {"correct", "incorrect", "format_error", "transient_failure", "permanent_failure"}
)


@dataclass(frozen=True)
class RewardAttempt:
    task_id: str
    candidate_index: int
    attempt: int
    status: str
    reward: float
    message: str

    def __post_init__(self):
        if self.candidate_index < 0 or self.attempt < 1:
            raise VerifiableRewardError("attempt coordinates are invalid")
        if self.status not in _STATUSES:
            raise VerifiableRewardError("attempt status is invalid")
        if not math.isfinite(self.reward) or not 0 <= self.reward <= 1:
            raise VerifiableRewardError("attempt reward must be finite in [0, 1]")
        if self.status.endswith("failure") and self.reward != 0:
            raise VerifiableRewardError("failed attempts cannot receive reward")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)
