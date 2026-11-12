"""Deterministic verifier dispatch with explicit format outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from .verifiable_reward_answer import extract_final_answer
from .verifiable_reward_contract import VerifiableRewardError
from .verifiable_reward_exact import verify_exact
from .verifiable_reward_json import verify_json
from .verifiable_reward_numeric import verify_numeric
from .verifiable_reward_task import VerifiableTask


@dataclass(frozen=True)
class VerificationOutcome:
    format_valid: bool
    correct: bool
    message: str


_VERIFIERS = {"exact": verify_exact, "numeric": verify_numeric, "json": verify_json}


def verify_response(
    task: VerifiableTask, response: str, *, max_bytes: int
) -> VerificationOutcome:
    try:
        answer = extract_final_answer(response, max_bytes=max_bytes)
        correct = _VERIFIERS[task.verifier](answer, task.expected, dict(task.options))
    except VerifiableRewardError as exc:
        return VerificationOutcome(False, False, str(exc))
    return VerificationOutcome(True, correct, "correct" if correct else "incorrect")
