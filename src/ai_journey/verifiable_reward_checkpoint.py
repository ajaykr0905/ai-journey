"""Sealed progress checkpoints for group-boundary recovery."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Mapping

from .verifiable_reward_contract import VerifiableRewardError

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class RewardCheckpoint:
    task_set_sha256: str
    next_group_index: int
    attempts_sha256: str

    def __post_init__(self):
        if not _SHA256.fullmatch(self.task_set_sha256) or not _SHA256.fullmatch(
            self.attempts_sha256
        ):
            raise VerifiableRewardError("checkpoint digests must be lowercase SHA-256")
        if self.next_group_index < 0:
            raise VerifiableRewardError("next_group_index must be non-negative")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RewardCheckpoint":
        if set(value) != {"task_set_sha256", "next_group_index", "attempts_sha256"}:
            raise VerifiableRewardError("checkpoint fields mismatch")
        if isinstance(value["next_group_index"], bool) or not isinstance(
            value["next_group_index"], int
        ):
            raise VerifiableRewardError("next_group_index must be an integer")
        return cls(
            str(value["task_set_sha256"]),
            value["next_group_index"],
            str(value["attempts_sha256"]),
        )
