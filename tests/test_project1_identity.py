from dataclasses import replace

import pytest

from ai_journey.project1_contract import Project1RunContract
from ai_journey.project1_identity import run_id


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
