"""Strict task records for public-safe verifiable rewards."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping

from .verifiable_reward_contract import VerifiableRewardError

_TASK_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_VERIFIERS = frozenset({"exact", "numeric", "json"})


def _json_value(value: object, name: str) -> object:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise VerifiableRewardError(f"{name} must be finite JSON") from exc
    if len(encoded.encode()) > 16_384:
        raise VerifiableRewardError(f"{name} exceeds 16384 bytes")
    return value


@dataclass(frozen=True)
class VerifiableTask:
    task_id: str
    prompt: str
    verifier: str
    expected: object
    options: Mapping[str, object]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "VerifiableTask":
        expected_fields = {"task_id", "prompt", "verifier", "expected", "options"}
        if set(value) != expected_fields:
            raise VerifiableRewardError("task fields mismatch")
        task_id = value["task_id"]
        prompt = value["prompt"]
        verifier = value["verifier"]
        options = value["options"]
        if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
            raise VerifiableRewardError("task_id is not canonical")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or len(prompt.encode()) > 4096
        ):
            raise VerifiableRewardError("prompt must contain at most 4096 UTF-8 bytes")
        if verifier not in _VERIFIERS:
            raise VerifiableRewardError("verifier is unsupported")
        if not isinstance(options, Mapping) or not all(
            isinstance(key, str) for key in options
        ):
            raise VerifiableRewardError("options must be a string-keyed mapping")
        return cls(
            task_id=task_id,
            prompt=prompt,
            verifier=verifier,
            expected=_json_value(value["expected"], "expected"),
            options=dict(_json_value(dict(options), "options")),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "verifier": self.verifier,
            "expected": self.expected,
            "options": dict(self.options),
        }
