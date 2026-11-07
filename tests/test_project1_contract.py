from dataclasses import replace

import pytest

from ai_journey.project1_contract import Project1RunContract, contract_from_mapping


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


def test_run_contract_has_stable_versioned_identity():
    first = contract()

    assert first.to_mapping()["schema_version"] == 1
    assert first.fingerprint() == contract().fingerprint()
    assert first.fingerprint() != replace(first, max_parameters=99_999).fingerprint()


def test_run_contract_round_trips_without_ignoring_fields():
    payload = contract().to_mapping()

    assert contract_from_mapping(payload) == contract()
    with pytest.raises(ValueError, match="fields"):
        contract_from_mapping({**payload, "ignored": True})
    with pytest.raises(ValueError, match="version"):
        contract_from_mapping({**payload, "schema_version": 2})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_name", "Day 64", "lowercase slug"),
        ("spec_sha256", "A" * 64, "SHA-256"),
        ("source_sha256", "short", "SHA-256"),
        ("max_source_bytes", 0, "positive"),
        ("max_parameters", True, "integer"),
        ("telemetry_interval_steps", -1, "positive"),
        ("checkpoint_interval_steps", 1.5, "integer"),
    ],
)
def test_run_contract_rejects_ambiguous_or_unbounded_controls(field, value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        replace(contract(), **{field: value})
