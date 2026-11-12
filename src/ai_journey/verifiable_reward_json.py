"""Strict structured-answer verification with duplicate-key rejection."""

from __future__ import annotations

import json

from .verifiable_reward_contract import VerifiableRewardError


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise VerifiableRewardError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def verify_json(answer: str, expected: object, options: object) -> bool:
    if options != {}:
        raise VerifiableRewardError("json verifier does not accept options")
    try:
        actual = json.loads(
            answer,
            object_pairs_hook=_unique_object,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
        )
    except VerifiableRewardError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError, RecursionError) as exc:
        raise VerifiableRewardError("answer must be strict finite JSON") from exc
    return actual == expected and type(actual) is type(expected)
