"""Hash-chained attempt journal for tamper-evident recovery evidence."""

from __future__ import annotations

from .verifiable_reward_attempt import RewardAttempt
from .verifiable_reward_identity import sha256_json

_GENESIS = "0" * 64


def build_attempt_journal(
    attempts: tuple[RewardAttempt, ...],
) -> list[dict[str, object]]:
    events = []
    previous = _GENESIS
    for sequence, attempt in enumerate(attempts):
        body = {
            "sequence": sequence,
            "previous_sha256": previous,
            "attempt": attempt.to_mapping(),
        }
        digest = sha256_json(body)
        event = {**body, "sha256": digest}
        events.append(event)
        previous = digest
    return events


def verify_attempt_journal(events: list[dict[str, object]]) -> bool:
    previous = _GENESIS
    for sequence, event in enumerate(events):
        if set(event) != {"sequence", "previous_sha256", "attempt", "sha256"}:
            return False
        body = {key: event[key] for key in ("sequence", "previous_sha256", "attempt")}
        if event["sequence"] != sequence or event["previous_sha256"] != previous:
            return False
        if event["sha256"] != sha256_json(body):
            return False
        previous = str(event["sha256"])
    return True
