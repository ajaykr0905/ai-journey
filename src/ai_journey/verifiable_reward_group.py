"""Candidate-group admission for group-relative reward evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .verifiable_reward_contract import RewardRunConfig, VerifiableRewardError


@dataclass(frozen=True)
class CandidateGroup:
    task_id: str
    responses: tuple[str, ...]


def admit_candidate_group(
    task_id: str, responses: Iterable[str], config: RewardRunConfig
) -> CandidateGroup:
    values = tuple(responses)
    if len(values) != config.candidates_per_task:
        raise VerifiableRewardError("candidate count does not match run contract")
    if not all(isinstance(value, str) for value in values):
        raise VerifiableRewardError("candidate responses must be text")
    if any(len(value.encode()) > config.max_response_bytes for value in values):
        raise VerifiableRewardError("candidate response exceeds byte budget")
    return CandidateGroup(task_id=task_id, responses=values)
