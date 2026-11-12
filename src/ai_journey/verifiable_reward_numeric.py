"""Finite decimal verification without binary-float ambiguity."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from .verifiable_reward_contract import VerifiableRewardError


def _decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise VerifiableRewardError(f"{name} must be numeric")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise VerifiableRewardError(f"{name} must be decimal") from exc
    if not number.is_finite():
        raise VerifiableRewardError(f"{name} must be finite")
    return number


def verify_numeric(answer: str, expected: object, options: object) -> bool:
    if not isinstance(options, dict) or set(options) - {"absolute_tolerance"}:
        raise VerifiableRewardError("numeric verifier options are unsupported")
    tolerance = _decimal(options.get("absolute_tolerance", "0"), "absolute_tolerance")
    if tolerance < 0:
        raise VerifiableRewardError("absolute_tolerance must be non-negative")
    return abs(_decimal(answer, "answer") - _decimal(expected, "expected")) <= tolerance
