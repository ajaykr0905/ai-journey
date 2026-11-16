"""Strict bounded controls for the telemetry-metrics lab.

The config declares the latency histogram bounds and the service-level
objectives that recording and alerting rules are generated from. Durations
are exact integers: the bucket bounds and the latency objective in
microseconds, so sub-millisecond loopback latencies get their own buckets,
and the other objectives in milliseconds, like the recovery labs' controls.
The one ratio is a float in (0, 1]. Missing, unknown and duplicate keys,
booleans posing as integers and non-finite numbers are rejected before any
evidence is read. The objectives are declared lab controls, not targets
taken from any external service or product.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .serving_recovery_config import (
    canonical_sha256,
    read_bounded_file,
    strict_json_loads,
)
from .telemetry_metrics_model import MAX_BUCKETS, TelemetryMetricsError

SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 16 * 1024
MAX_BUCKET_US = 60_000_000
MAX_OBJECTIVE_MS = 300_000
_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")


def _require_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise TelemetryMetricsError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _require_fields(payload: object, names: frozenset[str], section: str) -> dict:
    if type(payload) is not dict or set(payload) != names:
        raise TelemetryMetricsError(f"{section} fields do not match schema")
    return payload


@dataclass(frozen=True)
class ServingObjectives:
    """Declared objectives for one serving kill-and-recover run.

    Each duration is the largest value that still meets its objective. The
    latency objective asks that at least ``post_fault_within_latency_ratio``
    of the successes served by a post-fault lifetime took at most
    ``post_fault_latency_us``, which must be a histogram bucket bound so the
    rule can read it from one cumulative bucket.
    """

    detection_latency_ms: int
    restart_latency_ms: int
    unavailability_ms: int
    time_to_slo_recovery_ms: int
    post_fault_latency_us: int
    post_fault_within_latency_ratio: float

    def __post_init__(self) -> None:
        for name in (
            "detection_latency_ms",
            "restart_latency_ms",
            "unavailability_ms",
            "time_to_slo_recovery_ms",
        ):
            _require_int(getattr(self, name), name, 1, MAX_OBJECTIVE_MS)
        _require_int(
            self.post_fault_latency_us, "post_fault_latency_us", 1, MAX_BUCKET_US
        )
        ratio = self.post_fault_within_latency_ratio
        if type(ratio) is not float:
            raise TypeError("post_fault_within_latency_ratio must be a float")
        if not (math.isfinite(ratio) and 0 < ratio <= 1):
            raise TelemetryMetricsError(
                "post_fault_within_latency_ratio must be in (0, 1]"
            )


@dataclass(frozen=True)
class WorkerObjectives:
    """Declared largest detection and full recovery times for a worker run."""

    detection_ms: int
    recovery_ms: int

    def __post_init__(self) -> None:
        _require_int(self.detection_ms, "detection_ms", 1, MAX_OBJECTIVE_MS)
        _require_int(self.recovery_ms, "recovery_ms", 1, MAX_OBJECTIVE_MS)


@dataclass(frozen=True)
class TelemetryMetricsConfig:
    """Histogram bounds and objectives for one telemetry-metrics export."""

    experiment_name: str
    latency_buckets_us: tuple[int, ...]
    serving: ServingObjectives
    worker: WorkerObjectives

    def __post_init__(self) -> None:
        if (
            type(self.experiment_name) is not str
            or _NAME.fullmatch(self.experiment_name) is None
        ):
            raise TelemetryMetricsError("experiment_name must be a lowercase slug")
        buckets = self.latency_buckets_us
        if type(buckets) is not tuple or not 1 <= len(buckets) <= MAX_BUCKETS:
            raise TelemetryMetricsError(
                f"latency_buckets_us needs 1 to {MAX_BUCKETS} bounds"
            )
        for bound in buckets:
            _require_int(bound, "latency bucket", 1, MAX_BUCKET_US)
        if any(left >= right for left, right in zip(buckets, buckets[1:])):
            raise TelemetryMetricsError("latency_buckets_us must strictly increase")
        if not isinstance(self.serving, ServingObjectives) or not isinstance(
            self.worker, WorkerObjectives
        ):
            raise TypeError("objectives must be ServingObjectives and WorkerObjectives")
        if self.serving.post_fault_latency_us not in buckets:
            raise TelemetryMetricsError(
                "post_fault_latency_us must be one of latency_buckets_us"
            )

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical versioned mapping."""

        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_name": self.experiment_name,
            "latency_buckets_us": list(self.latency_buckets_us),
            "serving_objectives": {
                field.name: getattr(self.serving, field.name)
                for field in fields(ServingObjectives)
            },
            "worker_objectives": {
                field.name: getattr(self.worker, field.name)
                for field in fields(WorkerObjectives)
            },
        }

    def fingerprint(self) -> str:
        """Bind every bound and objective to one stable digest."""

        return canonical_sha256(self.to_mapping())


_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_name",
        "latency_buckets_us",
        "serving_objectives",
        "worker_objectives",
    }
)
_SERVING_FIELDS = frozenset(field.name for field in fields(ServingObjectives))
_WORKER_FIELDS = frozenset(field.name for field in fields(WorkerObjectives))


def config_from_mapping(payload: object) -> TelemetryMetricsConfig:
    """Parse controls without ignoring missing or unknown fields at any level."""

    payload = _require_fields(payload, _FIELDS, "telemetry metrics config")
    version = payload["schema_version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise TelemetryMetricsError("telemetry metrics config schema is unsupported")
    buckets = payload["latency_buckets_us"]
    if type(buckets) is not list:
        raise TelemetryMetricsError("latency_buckets_us must be a list")
    serving = _require_fields(
        payload["serving_objectives"], _SERVING_FIELDS, "serving_objectives"
    )
    worker = _require_fields(
        payload["worker_objectives"], _WORKER_FIELDS, "worker_objectives"
    )
    return TelemetryMetricsConfig(
        experiment_name=payload["experiment_name"],
        latency_buckets_us=tuple(buckets),
        serving=ServingObjectives(**serving),
        worker=WorkerObjectives(**worker),
    )


def load_telemetry_metrics_config(path: Path) -> TelemetryMetricsConfig:
    """Load bounded strict JSON controls from a regular non-symlink file."""

    try:
        data = read_bounded_file(path, max_bytes=MAX_CONFIG_BYTES, name="config")
        payload = strict_json_loads(data, name="config")
    except ValueError as error:
        raise TelemetryMetricsError(str(error)) from error
    return config_from_mapping(payload)
