"""Strict pinned controls for the Day 73 telemetry traces and dashboards lab.

The config declares the instant that each exported run's first lifecycle
record is mapped to. The labs record only ``time.monotonic_ns``, so that
anchor is a declared value, not an observation: it makes OTLP timestamps
reproducible while every duration stays a measured monotonic difference. The
anchor must be a whole second and is restated as UTC text that has to agree
with it. Missing, unknown and duplicate fields and non-finite numbers are
rejected.
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
_FIELDS = frozenset({"schema_version", "name", "time_mapping"})
_TIME_FIELDS = frozenset({"anchor_unix_nano", "anchor_utc", "note"})


def utc_text(unix_nano: int) -> str:
    """Render a whole-second Unix nanosecond instant as ``YYYY-MM-DDTHH:MM:SSZ``."""

    moment = datetime.fromtimestamp(unix_nano // 1_000_000_000, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class TelemetryConfig:
    """Declared, reproducible controls for trace export and dashboards."""

    name: str
    anchor_unix_nano: int
    anchor_utc: str
    anchor_note: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or _NAME.fullmatch(self.name) is None:
            raise TelemetryError("name must be a lowercase slug")
        anchor = self.anchor_unix_nano
        if type(anchor) is not int or not 1_000_000_000 <= anchor <= 2**63 - 1:
            raise TelemetryError("anchor_unix_nano must be in [1e9, 2**63 - 1]")
        if anchor % 1_000_000_000:
            raise TelemetryError("anchor_unix_nano must be a whole second")
        if self.anchor_utc != utc_text(anchor):
            raise TelemetryError("anchor_utc must restate anchor_unix_nano")
        if (
            type(self.anchor_note) is not str
            or _NOTE.fullmatch(self.anchor_note) is None
        ):
            raise TelemetryError("time_mapping note must be printable ASCII")

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
    time_mapping = payload["time_mapping"]
    if type(time_mapping) is not dict or set(time_mapping) != _TIME_FIELDS:
        raise TelemetryError("time_mapping fields do not match schema")
    return TelemetryConfig(
        name=payload["name"],
        anchor_unix_nano=time_mapping["anchor_unix_nano"],
        anchor_utc=time_mapping["anchor_utc"],
        anchor_note=time_mapping["note"],
    )


def load_telemetry_config(path: Path) -> TelemetryConfig:
    """Load bounded strict JSON controls from a regular file."""

    data = read_bounded_file(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    return config_from_mapping(strict_json_loads(data, label="config"))
