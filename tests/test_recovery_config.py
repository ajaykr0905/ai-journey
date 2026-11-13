import pytest

from ai_journey.recovery_config import (
    RecoveryExperimentConfig,
    config_from_mapping,
)


def valid_config():
    return RecoveryExperimentConfig(
        experiment_name="day-70-worker-recovery",
        work_items=20,
        checkpoint_interval=4,
        fault_after_items=11,
        heartbeat_interval_items=2,
        heartbeat_timeout_ms=500,
        worker_timeout_ms=5000,
    )


def test_config_round_trips_and_has_stable_identity():
    config = valid_config()

    assert config_from_mapping(config.to_mapping()) == config
    assert len(config.fingerprint()) == 64
    assert config.fingerprint() == valid_config().fingerprint()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("experiment_name", "Day 70", "lowercase slug"),
        ("work_items", True, "integer"),
        ("checkpoint_interval", 20, "precede completion"),
        ("fault_after_items", 4, "follow a checkpoint"),
        ("fault_after_items", 20, "precede completion"),
        ("heartbeat_interval_items", 5, "cannot exceed"),
        ("heartbeat_timeout_ms", 5000, "lower than"),
    ],
)
def test_config_rejects_unbounded_or_incoherent_controls(field, value, message):
    values = valid_config().to_mapping()
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        config_from_mapping(values)


def test_config_parser_rejects_unknown_fields_and_versions():
    payload = valid_config().to_mapping()
    with pytest.raises(ValueError, match="fields"):
        config_from_mapping({**payload, "ignored": True})
    with pytest.raises(ValueError, match="version"):
        config_from_mapping({**payload, "schema_version": 2})
