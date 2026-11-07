from dataclasses import replace

import pytest

from ai_journey.project1_contract import Project1RunContract
from ai_journey.project1_identity import attempt_id, run_id


def contract() -> Project1RunContract:
    return Project1RunContract(
        run_name="day-64-project1",
        spec_sha256="a" * 64,
        source_sha256="b" * 64,
        max_source_bytes=1024,
        max_parameters=100_000,
        telemetry_interval_steps=1,
        checkpoint_interval_steps=2,
    )


def test_run_identity_is_readable_stable_and_control_bound():
    value = run_id(contract())

    assert value.startswith("day-64-project1-")
    assert value == run_id(contract())
    assert value != run_id(replace(contract(), max_parameters=99_999))


def test_run_identity_requires_a_validated_contract():
    with pytest.raises(TypeError, match="Project1RunContract"):
        run_id({})


def test_attempt_identity_binds_attempt_number_and_parent_checkpoint():
    identity = run_id(contract())

    fresh = attempt_id(identity, 1)
    resumed = attempt_id(identity, 2, parent_checkpoint_sha256="c" * 64)
    assert fresh == attempt_id(identity, 1)
    assert fresh.startswith("attempt-1-")
    assert resumed.startswith("attempt-2-")
    assert fresh != resumed


@pytest.mark.parametrize(
    ("identity", "number", "parent", "message"),
    [
        ("bad", 1, None, "run_identity"),
        (run_id(contract()), True, None, "integer"),
        (run_id(contract()), 0, None, "positive"),
        (run_id(contract()), 2, "bad", "SHA-256"),
    ],
)
def test_attempt_identity_rejects_ambiguous_lineage(identity, number, parent, message):
    with pytest.raises((TypeError, ValueError), match=message):
        attempt_id(identity, number, parent_checkpoint_sha256=parent)
