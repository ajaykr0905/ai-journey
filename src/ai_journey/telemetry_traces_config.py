"""Strict pinned controls for the Day 73 telemetry traces and dashboards lab.

``time_mapping`` declares the instant that each exported run's first lifecycle
record is mapped to. The labs record only ``time.monotonic_ns``, so that
anchor is a declared value, not an observation: it makes OTLP timestamps
reproducible while every duration stays a measured monotonic difference. The
anchor must be a whole second and is restated as UTC text that has to agree
with it. ``dashboard`` pins the generated Grafana dashboard's identity, time
range, refresh interval and latency quantiles. Missing, unknown and duplicate
fields and non-finite numbers are rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .telemetry_traces_io import (
    TelemetryError,
    canonical_sha256,
    read_bounded_file,
    strict_json_loads,
)

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 16 * 1024
_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_NOTE = re.compile(r"[ -~]{1,1024}")
# Grafana dashboard UIDs are at most 40 URL-safe characters.
_UID = re.compile(r"[A-Za-z0-9_-]{1,40}")
_TITLE = re.compile(r"[!-~](?:[ -~]{0,126}[!-~])?")
_TIME_FROM = re.compile(r"now-[1-9][0-9]{0,3}[mhdw]")
_REFRESH = re.compile(r"(?:[1-9][0-9]{0,3}[smhd])?")
_FIELDS = frozenset({"schema_version", "name", "time_mapping", "dashboard"})
_TIME_FIELDS = frozenset({"anchor_unix_nano", "anchor_utc", "note"})
_DASHBOARD_FIELDS = frozenset(
    {"uid", "title", "time_from", "refresh", "latency_quantiles"}
)


def utc_text(unix_nano: int) -> str:
    """Render a whole-second Unix nanosecond instant as ``YYYY-MM-DDTHH:MM:SSZ``."""

    moment = datetime.fromtimestamp(unix_nano // 1_000_000_000, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_text(value: object, pattern: re.Pattern[str], message: str) -> None:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise TelemetryError(message)


@dataclass(frozen=True)
class TelemetryConfig:
    """Declared, reproducible controls for trace export and dashboards."""

    name: str
    anchor_unix_nano: int
    anchor_utc: str
    anchor_note: str
    dashboard_uid: str
    dashboard_title: str
    dashboard_time_from: str
    dashboard_refresh: str
    latency_quantiles: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_text(self.name, _NAME, "name must be a lowercase slug")
        anchor = self.anchor_unix_nano
        if type(anchor) is not int or not 1_000_000_000 <= anchor <= 2**63 - 1:
            raise TelemetryError("anchor_unix_nano must be in [1e9, 2**63 - 1]")
        if anchor % 1_000_000_000:
            raise TelemetryError("anchor_unix_nano must be a whole second")
        if self.anchor_utc != utc_text(anchor):
            raise TelemetryError("anchor_utc must restate anchor_unix_nano")
        _require_text(
            self.anchor_note, _NOTE, "time_mapping note must be printable ASCII"
        )
        _require_text(self.dashboard_uid, _UID, "dashboard uid must be a Grafana UID")
        _require_text(
            self.dashboard_title, _TITLE, "dashboard title must be printable ASCII"
        )
        _require_text(
            self.dashboard_time_from, _TIME_FROM, "time_from must look like now-6h"
        )
        _require_text(
            self.dashboard_refresh, _REFRESH, "refresh must be empty or like 1m"
        )
        quantiles = self.latency_quantiles
        if (
            type(quantiles) is not tuple
            or not 1 <= len(quantiles) <= 5
            or any(type(q) is not float or not 0 < q < 1 for q in quantiles)
            or list(quantiles) != sorted(set(quantiles))
        ):
            raise TelemetryError(
                "latency_quantiles must be 1 to 5 increasing floats in (0, 1)"
            )

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical versioned mapping."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "time_mapping": {
                "anchor_unix_nano": self.anchor_unix_nano,
                "anchor_utc": self.anchor_utc,
                "note": self.anchor_note,
            },
            "dashboard": {
                "uid": self.dashboard_uid,
                "title": self.dashboard_title,
                "time_from": self.dashboard_time_from,
                "refresh": self.dashboard_refresh,
                "latency_quantiles": list(self.latency_quantiles),
            },
        }

    def fingerprint(self) -> str:
        """Bind every control to one stable digest."""

        return canonical_sha256(self.to_mapping())


def config_from_mapping(payload: object) -> TelemetryConfig:
    """Parse controls without ignoring missing or unknown fields."""

    if type(payload) is not dict or set(payload) != _FIELDS:
        raise TelemetryError("telemetry config fields do not match schema")
    version = payload["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise TelemetryError("telemetry config schema is unsupported")
    time_mapping, dashboard = payload["time_mapping"], payload["dashboard"]
    if type(time_mapping) is not dict or set(time_mapping) != _TIME_FIELDS:
        raise TelemetryError("time_mapping fields do not match schema")
    if type(dashboard) is not dict or set(dashboard) != _DASHBOARD_FIELDS:
        raise TelemetryError("dashboard fields do not match schema")
    quantiles = dashboard["latency_quantiles"]
    if type(quantiles) is not list:
        raise TelemetryError("latency_quantiles must be a list")
    return TelemetryConfig(
        name=payload["name"],
        anchor_unix_nano=time_mapping["anchor_unix_nano"],
        anchor_utc=time_mapping["anchor_utc"],
        anchor_note=time_mapping["note"],
        dashboard_uid=dashboard["uid"],
        dashboard_title=dashboard["title"],
        dashboard_time_from=dashboard["time_from"],
        dashboard_refresh=dashboard["refresh"],
        latency_quantiles=tuple(quantiles),
    )


def load_telemetry_config(path: Path) -> TelemetryConfig:
    """Load bounded strict JSON controls from a regular file."""

    data = read_bounded_file(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    return config_from_mapping(strict_json_loads(data, label="config"))
