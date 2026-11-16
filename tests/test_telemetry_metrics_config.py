import json
import math

import pytest

from ai_journey.serving_recovery_config import canonical_json_bytes
from ai_journey.telemetry_metrics_config import (
    MAX_CONFIG_BYTES,
    TelemetryMetricsConfig,
    config_from_mapping,
    load_telemetry_metrics_config,
)
from ai_journey.telemetry_metrics_model import TelemetryMetricsError


def config_mapping(**changes):
    mapping = {
        "schema_version": 1,
        "experiment_name": "telemetry-metrics-test",
        "latency_buckets_us": [250, 500, 1000, 2500, 10_000, 250_000, 1_000_000],
        "serving_objectives": {
            "detection_latency_ms": 250,
            "restart_latency_ms": 2000,
            "unavailability_ms": 3000,
            "time_to_slo_recovery_ms": 5000,
            "post_fault_latency_us": 250_000,
            "post_fault_within_latency_ratio": 0.95,
        },
        "worker_objectives": {"detection_ms": 1000, "recovery_ms": 5000},
    }
    for key, value in changes.items():
        if key in ("serving_objectives", "worker_objectives") and type(value) is dict:
            mapping[key] = mapping[key] | value
        else:
            mapping[key] = value
    return mapping


def valid_config(**changes) -> TelemetryMetricsConfig:
    return config_from_mapping(config_mapping(**changes))


def test_config_round_trips_and_binds_every_control_to_its_digest():
    config = valid_config()

    assert config_from_mapping(config.to_mapping()) == config
    assert config.to_mapping() == config_mapping()
    assert config.latency_buckets_us[0] == 250
    assert config.serving.post_fault_within_latency_ratio == 0.95
    assert len(config.fingerprint()) == 64
    assert config.fingerprint() == valid_config().fingerprint()
    assert (
        config.fingerprint()
        != valid_config(worker_objectives={"recovery_ms": 5001}).fingerprint()
    )


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"schema_version": 2}, TelemetryMetricsError, "schema"),
        ({"schema_version": True}, TelemetryMetricsError, "schema"),
        ({"experiment_name": "Day 73"}, TelemetryMetricsError, "slug"),
        ({"latency_buckets_us": []}, TelemetryMetricsError, "1 to 32"),
        ({"latency_buckets_us": list(range(1, 34))}, TelemetryMetricsError, "1 to 32"),
        ({"latency_buckets_us": "250"}, TelemetryMetricsError, "must be a list"),
        (
            {"latency_buckets_us": [500, 250, 250_000]},
            TelemetryMetricsError,
            "strictly increase",
        ),
        (
            {"latency_buckets_us": [250, 250, 250_000]},
            TelemetryMetricsError,
            "strictly increase",
        ),
        ({"latency_buckets_us": [0, 250_000]}, TelemetryMetricsError, "latency bucket"),
        (
            {"latency_buckets_us": [250_000, 60_000_001]},
            TelemetryMetricsError,
            "latency bucket",
        ),
        ({"latency_buckets_us": [True, 250_000]}, TypeError, "integer"),
        ({"latency_buckets_us": [2.5e5]}, TypeError, "integer"),
        (
            {"latency_buckets_us": [250, 500]},
            TelemetryMetricsError,
            "one of latency_buckets_us",
        ),
        (
            {"serving_objectives": {"detection_latency_ms": 0}},
            TelemetryMetricsError,
            "detection_latency_ms",
        ),
        (
            {"serving_objectives": {"restart_latency_ms": 300_001}},
            TelemetryMetricsError,
            "restart_latency_ms",
        ),
        ({"serving_objectives": {"unavailability_ms": 1.0}}, TypeError, "integer"),
        (
            {"serving_objectives": {"post_fault_within_latency_ratio": 1}},
            TypeError,
            "float",
        ),
        (
            {"serving_objectives": {"post_fault_within_latency_ratio": 0.0}},
            TelemetryMetricsError,
            r"\(0, 1\]",
        ),
        (
            {"serving_objectives": {"post_fault_within_latency_ratio": 1.5}},
            TelemetryMetricsError,
            r"\(0, 1\]",
        ),
        (
            {"serving_objectives": {"post_fault_within_latency_ratio": math.nan}},
            TelemetryMetricsError,
            r"\(0, 1\]",
        ),
        ({"worker_objectives": {"recovery_ms": False}}, TypeError, "integer"),
        ({"worker_objectives": {"extra": 1}}, TelemetryMetricsError, "schema"),
        ({"serving_objectives": []}, TelemetryMetricsError, "schema"),
        ({"unknown": 1}, TelemetryMetricsError, "schema"),
    ],
)
def test_config_rejects_invalid_controls(changes, error, message):
    with pytest.raises(error, match=message):
        valid_config(**changes)


def test_config_rejects_missing_fields_at_every_level():
    for section, field in (
        (None, "worker_objectives"),
        ("serving_objectives", "post_fault_latency_us"),
        ("worker_objectives", "detection_ms"),
    ):
        mapping = config_mapping()
        del (mapping if section is None else mapping[section])[field]
        with pytest.raises(TelemetryMetricsError, match="schema"):
            config_from_mapping(mapping)


def test_loader_reads_bounded_strict_json_files(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(canonical_json_bytes(config_mapping()))
    assert load_telemetry_metrics_config(path) == valid_config()

    path.write_text(json.dumps(config_mapping(), indent=2))
    assert load_telemetry_metrics_config(path) == valid_config()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}')
    nan = tmp_path / "nan.json"
    nan.write_text(
        json.dumps(
            config_mapping(serving_objectives={"post_fault_within_latency_ratio": 1.0})
        ).replace("1.0", "NaN")
    )
    large = tmp_path / "large.json"
    large.write_bytes(b" " * (MAX_CONFIG_BYTES + 1))
    link = tmp_path / "link.json"
    link.symlink_to(path)
    for bad, message in (
        (duplicate, "duplicate JSON key"),
        (nan, "non-finite"),
        (large, "exceeds"),
        (link, "non-symlink"),
        (tmp_path / "missing.json", "readable regular"),
        (tmp_path, "regular"),
    ):
        with pytest.raises(TelemetryMetricsError, match=message):
            load_telemetry_metrics_config(bad)
