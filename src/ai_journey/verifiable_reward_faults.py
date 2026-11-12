"""Deterministic, declared verifier failure injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .verifiable_reward_contract import VerifiableRewardError


@dataclass(frozen=True)
class InjectedFault:
    task_id: str
    candidate_index: int
    attempt: int
    kind: str


class FaultPlan:
    def __init__(self, faults: Iterable[InjectedFault] = ()):
        entries = tuple(faults)
        keys = [(item.task_id, item.candidate_index, item.attempt) for item in entries]
        if len(keys) != len(set(keys)):
            raise VerifiableRewardError("fault coordinates must be unique")
        if any(item.kind not in {"transient", "permanent"} for item in entries):
            raise VerifiableRewardError("fault kind must be transient or permanent")
        if any(item.candidate_index < 0 or item.attempt < 1 for item in entries):
            raise VerifiableRewardError("fault coordinates must be non-negative")
        self._faults = dict(zip(keys, (item.kind for item in entries), strict=True))

    def kind_for(self, task_id: str, candidate_index: int, attempt: int) -> str | None:
        return self._faults.get((task_id, candidate_index, attempt))

    def to_records(self) -> list[Mapping[str, object]]:
        return [
            {
                "task_id": key[0],
                "candidate_index": key[1],
                "attempt": key[2],
                "kind": kind,
            }
            for key, kind in sorted(self._faults.items())
        ]
