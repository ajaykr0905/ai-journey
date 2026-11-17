import json
from pathlib import Path

import pytest

from ai_journey.telemetry_traces_config import (
    config_from_mapping,
    load_telemetry_config,
    utc_text,
)
from ai_journey.telemetry_traces_io import TelemetryError

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config" / "day-73-telemetry-traces.json"


def mapping(time=None, dashboard=None, **top):
    return {
        "schema_version": 1,
        "name": "telemetry-test",
        "time_mapping": {
            "anchor_unix_nano": 1_794_787_200_000_000_000,
            "anchor_utc": "2026-11-16T00:00:00Z",
            "note": "Declared test anchor.",
        }
        | (time or {}),
        "dashboard": {
            "uid": "telemetry-test",
            "title": "Telemetry test: local CPU lab",
            "time_from": "now-1h",
            "refresh": "",
            "latency_quantiles": [0.5, 0.99],
        }
        | (dashboard or {}),
    } | top


def test_pinned_config_declares_the_anchor_and_dashboard_controls():
    config = load_telemetry_config(PINNED)

    assert config.name == "day-73-telemetry-traces"
    assert config.anchor_unix_nano == 1_794_787_200_000_000_000
    assert config.anchor_utc == "2026-11-16T00:00:00Z"
    assert "synthetic" in config.anchor_note and "measured" in config.anchor_note
    assert config.dashboard_uid == "ai-journey-day-73-recovery"
    assert "local CPU lab" in config.dashboard_title
    assert config.latency_quantiles == (0.5, 0.95, 0.99)
    assert config.to_mapping() == json.loads(PINNED.read_bytes())
    assert config_from_mapping(config.to_mapping()) == config
    assert len(config.fingerprint()) == 64


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (mapping({"anchor_unix_nano": 1_794_787_200_000_000_001}), "whole second"),
        (mapping({"anchor_unix_nano": 2**63}), r"2\*\*63"),
        (mapping({"anchor_unix_nano": True}), "anchor_unix_nano"),
        (mapping({"anchor_unix_nano": 1.7947872e18}), "anchor_unix_nano"),
        (mapping({"anchor_utc": "2026-11-16T00:00:01Z"}), "restate"),
        (mapping({"note": "tab\tseparated"}), "printable"),
        (mapping({"note": ""}), "printable"),
        (mapping(dashboard={"uid": "a" * 41}), "Grafana UID"),
        (mapping(dashboard={"uid": "has space"}), "Grafana UID"),
        (mapping(dashboard={"title": ""}), "title"),
        (mapping(dashboard={"time_from": "yesterday"}), "now-6h"),
        (mapping(dashboard={"refresh": "5"}), "refresh"),
        (mapping(dashboard={"latency_quantiles": []}), "quantiles"),
        (mapping(dashboard={"latency_quantiles": [0.99, 0.5]}), "increasing"),
        (mapping(dashboard={"latency_quantiles": [0.5, 0.5]}), "increasing"),
        (mapping(dashboard={"latency_quantiles": [1.0]}), r"\(0, 1\)"),
        (mapping(dashboard={"latency_quantiles": [1]}), "floats"),
        (mapping(dashboard={"latency_quantiles": 0.5}), "list"),
        (mapping(dashboard={"panels": []}), "dashboard fields"),
        (mapping(name="Day 73"), "slug"),
        (mapping(schema_version=2), "schema is unsupported"),
        (mapping(extra=1), "fields do not match"),
        (mapping(time_mapping={"anchor_unix_nano": 1}), "time_mapping"),
    ],
)
def test_invalid_controls_are_rejected(payload, message):
    with pytest.raises(TelemetryError, match=message):
        config_from_mapping(payload)


def test_config_files_are_strict_bounded_json(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"name": "a", "name": "b"}')
    with pytest.raises(TelemetryError, match="duplicate JSON key: name"):
        load_telemetry_config(duplicate)
    large = tmp_path / "large.json"
    large.write_bytes(b" " * (16 * 1024 + 1))
    with pytest.raises(TelemetryError, match="bound"):
        load_telemetry_config(large)
    assert utc_text(0) == "1970-01-01T00:00:00Z"
    earliest = config_from_mapping(
        mapping({"anchor_unix_nano": 10**9, "anchor_utc": "1970-01-01T00:00:01Z"})
    )
    assert earliest.anchor_unix_nano == 10**9
