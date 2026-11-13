import pytest
from test_recovery_config import valid_config

from ai_journey.recovery_identity import recovery_attempt_id, recovery_run_id


def test_run_identity_is_readable_and_bound_to_config():
    first = recovery_run_id(valid_config())

    assert first.startswith("day-70-worker-recovery-")
    assert len(first.rsplit("-", 1)[-1]) == 16
    assert first == recovery_run_id(valid_config())


def test_attempt_identity_binds_number_and_parent_checkpoint():
    run = recovery_run_id(valid_config())
    first = recovery_attempt_id(run, 1)
    resumed = recovery_attempt_id(run, 2, parent_checkpoint_sha256="a" * 64)

    assert first.startswith("attempt-1-")
    assert resumed.startswith("attempt-2-")
    assert first != resumed
    assert resumed != recovery_attempt_id(run, 2, parent_checkpoint_sha256="b" * 64)


@pytest.mark.parametrize(
    ("run", "number", "parent", "message"),
    [
        ("bad", 1, None, "run_identity"),
        ("day-70-0123456789abcdef", True, None, "integer"),
        ("day-70-0123456789abcdef", 0, None, "positive"),
        ("day-70-0123456789abcdef", 2, "bad", "SHA-256"),
    ],
)
def test_attempt_identity_rejects_ambiguous_values(run, number, parent, message):
    with pytest.raises((TypeError, ValueError), match=message):
        recovery_attempt_id(run, number, parent_checkpoint_sha256=parent)
