"""Bounded extraction of a candidate's declared final answer."""

from __future__ import annotations

from .verifiable_reward_contract import VerifiableRewardError


def extract_final_answer(response: str, *, max_bytes: int) -> str:
    if not isinstance(response, str):
        raise VerifiableRewardError("response must be text")
    if len(response.encode("utf-8")) > max_bytes:
        raise VerifiableRewardError("response exceeds byte budget")
    lines = response.splitlines()
    markers = [line[6:].strip() for line in lines if line.startswith("FINAL:")]
    if len(markers) != 1 or not markers[0]:
        raise VerifiableRewardError(
            "response must contain exactly one non-empty FINAL: line"
        )
    return markers[0]
