"""Exact-answer verification with explicit normalization controls."""

from __future__ import annotations

import unicodedata

from .verifiable_reward_contract import VerifiableRewardError


def verify_exact(answer: str, expected: object, options: object) -> bool:
    if not isinstance(expected, str) or not isinstance(options, dict):
        raise VerifiableRewardError(
            "exact verifier requires text expected and object options"
        )
    allowed = {"case_sensitive", "collapse_whitespace"}
    if set(options) - allowed:
        raise VerifiableRewardError("exact verifier options are unsupported")
    case_sensitive = options.get("case_sensitive", True)
    collapse = options.get("collapse_whitespace", False)
    if not isinstance(case_sensitive, bool) or not isinstance(collapse, bool):
        raise VerifiableRewardError("exact verifier options must be booleans")

    def normalize(value: str) -> str:
        value = unicodedata.normalize("NFC", value).strip()
        if collapse:
            value = " ".join(value.split())
        return value if case_sensitive else value.casefold()

    return normalize(answer) == normalize(expected)
