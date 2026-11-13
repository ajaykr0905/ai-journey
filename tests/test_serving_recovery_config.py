import json

import pytest

from ai_journey.serving_recovery_config import (
    ServingRecoveryConfig,
    ServingRecoveryError,
    canonical_json_bytes,
    config_from_mapping,
    load_serving_recovery_config,
    read_bounded_file,
    strict_json_loads,
)


def valid_config(**changes):
    values = {
        "experiment_name": "serving-recovery-test",
        "seed": 70,
        "requests": 24,
        "concurrency": 2,
        "max_new_tokens": 8,
        "fault_after_acknowledged": 6,
        "probe_interval_ms": 10,
        "probe_timeout_ms": 200,
        "detection_timeout_ms": 1000,
        "readiness_timeout_ms": 3000,
        "restart_limit": 2,
        "restart_backoff_ms": 20,
        "request_deadline_ms": 1000,
        "retry_budget": 14,
        "retry_backoff_ms": 10,
        "max_retry_backoff_ms": 500,
        "slo_latency_ms": 200,
        "slo_consecutive_successes": 4,
        "max_run_ms": 60_000,
        "shutdown_timeout_ms": 2000,
    }
    return ServingRecoveryConfig(**(values | changes))


def test_config_round_trips_and_binds_every_control_to_its_digest():
    config = valid_config()

    assert config_from_mapping(config.to_mapping()) == config
    assert config.to_mapping()["schema_version"] == 1
    assert len(config.fingerprint()) == 64
    assert config.fingerprint() == valid_config().fingerprint()
    assert config.fingerprint() != valid_config(seed=71).fingerprint()


def test_backoff_schedules_are_capped_and_cover_the_recovery_bound():
    config = valid_config()

    assert [config.retry_delay_ms(retry) for retry in (1, 2, 3, 6, 7, 14)] == [
        10,
        20,
        40,
        320,
        500,
        500,
    ]
    assert [config.restart_delay_ms(restart) for restart in (1, 2)] == [20, 40]
    assert config.retry_window_ms == 4630
    assert config.recovery_bound_ms == 4020
    with pytest.raises(ValueError, match="retry"):
        config.retry_delay_ms(15)
    with pytest.raises(ValueError, match="restart"):
        config.restart_delay_ms(3)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"experiment_name": "Serving Lab"}, "slug"),
        ({"requests": True}, "integer"),
        ({"concurrency": 5}, "concurrency"),
        ({"requests": 2, "concurrency": 3}, "concurrency"),
        ({"fault_after_acknowledged": 24}, "precede the final request"),
        ({"fault_after_acknowledged": 20}, "too few post-fault"),
        ({"probe_interval_ms": 1000}, "probe_interval_ms"),
        ({"probe_timeout_ms": 1001}, "probe_timeout_ms"),
        ({"slo_latency_ms": 1001}, "slo_latency_ms"),
        ({"retry_backoff_ms": 600}, "max_retry_backoff_ms"),
        ({"retry_budget": 4}, "outlast"),
        ({"max_run_ms": 15_000}, "worst-case retries"),
        ({"shutdown_timeout_ms": 10_000, "max_run_ms": 20_000}, "supervisor"),
        ({"seed": -1}, "seed"),
    ],
)
def test_config_rejects_ambiguous_or_incoherent_controls(changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        valid_config(**changes)


def test_config_parser_rejects_unknown_missing_and_versioned_fields():
    payload = valid_config().to_mapping()

    with pytest.raises(ValueError, match="fields"):
        config_from_mapping(payload | {"ignored": 1})
    with pytest.raises(ValueError, match="fields"):
        config_from_mapping({k: v for k, v in payload.items() if k != "seed"})
    with pytest.raises(ValueError, match="schema"):
        config_from_mapping(payload | {"schema_version": 2})
    with pytest.raises(ValueError, match="schema"):
        config_from_mapping(payload | {"schema_version": True})


def test_loader_rejects_duplicate_keys_and_non_finite_numbers(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_config().to_mapping()))
    assert load_serving_recovery_config(path) == valid_config()

    path.write_text('{"seed": 1, "seed": 2}')
    with pytest.raises(ServingRecoveryError, match="duplicate JSON key"):
        load_serving_recovery_config(path)
    for constant in ("NaN", "Infinity", "1e999"):
        path.write_text('{"seed": %s}' % constant)
        with pytest.raises(ServingRecoveryError, match="strict JSON"):
            load_serving_recovery_config(path)
    path.write_bytes(b'{"seed": "\xff"}')
    with pytest.raises(ServingRecoveryError, match="UTF-8"):
        load_serving_recovery_config(path)


def test_bounded_reads_refuse_aliases_directories_and_oversized_files(tmp_path):
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)

    assert read_bounded_file(target, max_bytes=2, name="input") == b"{}"
    with pytest.raises(ServingRecoveryError, match="non-symlink"):
        read_bounded_file(alias, max_bytes=2, name="input")
    with pytest.raises(ServingRecoveryError, match="regular file"):
        read_bounded_file(tmp_path, max_bytes=2, name="input")
    with pytest.raises(ServingRecoveryError, match="2-byte bound"):
        target.write_bytes(b"{ }")
        read_bounded_file(target, max_bytes=2, name="input")
    with pytest.raises(TypeError, match="pathlib"):
        read_bounded_file(str(target), max_bytes=2, name="input")


def test_canonical_json_is_sorted_compact_ascii_and_strictly_decoded():
    encoded = canonical_json_bytes({"b": [1, 2], "a": "é"})

    assert encoded == b'{"a":"\\u00e9","b":[1,2]}'
    assert strict_json_loads(encoded, name="value") == {"a": "é", "b": [1, 2]}
    with pytest.raises(ValueError):
        canonical_json_bytes({"x": float("nan")})
    with pytest.raises(TypeError, match="bytes"):
        strict_json_loads("{}", name="value")
