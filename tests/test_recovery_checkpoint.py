import copy

import pytest

from ai_journey.recovery_checkpoint import (
    RecoveryCheckpoint,
    checkpoint_from_mapping,
)


def checkpoint():
    return RecoveryCheckpoint(
        run_id="day-70-worker-0123456789abcdef",
        attempt_id="attempt-1-0123456789abcdef",
        completed_items=8,
        workload_sha256="a" * 64,
        accumulator_sha256="b" * 64,
    )


def test_checkpoint_is_canonical_and_self_sealed():
    payload = checkpoint().to_mapping()

    assert checkpoint_from_mapping(payload) == checkpoint()
    assert len(payload["checkpoint_sha256"]) == 64
    assert payload == checkpoint().to_mapping()


def test_checkpoint_verifier_rejects_tampering_and_unknown_fields():
    payload = checkpoint().to_mapping()
    changed = copy.deepcopy(payload)
    changed["completed_items"] = 9

    with pytest.raises(ValueError, match="digest"):
        checkpoint_from_mapping(changed)
    with pytest.raises(ValueError, match="fields"):
        checkpoint_from_mapping({**payload, "ignored": True})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", "bad", "run_id"),
        ("attempt_id", "bad", "attempt_id"),
        ("completed_items", True, "integer"),
        ("completed_items", -1, "non-negative"),
        ("workload_sha256", "bad", "SHA-256"),
        ("accumulator_sha256", "bad", "SHA-256"),
    ],
)
def test_checkpoint_rejects_invalid_progress_or_identities(field, value, message):
    values = checkpoint().__dict__ | {field: value}
    with pytest.raises((TypeError, ValueError), match=message):
        RecoveryCheckpoint(**values)
