"""Pinned thresholds of the Day 75 bottleneck rule for the naive Metal matmul.

The rule names the top bottleneck of each measured point from timings and the
measured Day 74 ceilings, so each name is only as good as its thresholds.
Every threshold lives here, in one strict config with one SHA-256 digest, and
a report binds to that digest. No threshold has a default in code, so a
report cannot get a name from a value that the config does not state.

Parsing rejects unknown and missing fields, integers where a fraction is due,
booleans posing as numbers, and values outside their range. Files go through
the bounded strict reader of ``memory_access_report_io``, which also rejects
duplicate keys. Nothing here imports torch.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .matmul_model_traffic import TRAFFIC_MODELS
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 16 * 1024
MAX_AGREEMENT_RATIO = 10.0

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_FIELDS = {"schema_version", "name", "rule", "mapping"}
_RULE_FIELDS = {
    "intensity_model",
    "dispatch_overhead_share_min",
    "near_roof_share_min",
    "timing_tolerance",
    "gpu_agreement_max_ratio",
    "roof_excess_tolerance",
    "require_fma_scales_with_work",
}
_MAPPING_FIELDS = {"slower_ratio_min"}


class MatmulBottleneckError(ValueError):
    """Raised when bottleneck inputs are unreadable, invalid or do not pair."""


class MatmulBottleneckVerificationError(MatmulBottleneckError):
    """Raised when a bottleneck report, or one of its sources, fails its checks."""


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise MatmulBottleneckError(f"{label} fields do not match the schema")
    return payload


def require_fraction(
    value: object, name: str, low: float, high: float, *, open_low: bool = False
) -> float:
    """Return a finite JSON float in range; an integer such as 1 is refused.

    A float-only rule keeps ``0.5`` and ``1`` from meaning one threshold in a
    file and another in a digest.
    """

    if type(value) is not float or not math.isfinite(value):
        raise MatmulBottleneckError(f"{name} must be a finite float")
    if value > high or value < low or (open_low and value == low):
        bracket = "(" if open_low else "["
        raise MatmulBottleneckError(f"{name} must lie in {bracket}{low}, {high}]")
    return value


@dataclass(frozen=True)
class BottleneckRule:
    """The thresholds of the ordered rule, as fractions or ratios.

    ``dispatch_overhead_share_min``: a call is dispatch-bound when at least
    this share of its one-call host time lies outside the steady-state time
    per call.
    ``near_roof_share_min``: a rate at or above this share of the lower
    measured roof sits on that roof.
    ``timing_tolerance``: how far the one-call GPU time may exceed the host
    time, and the steady-state time may exceed the host time, as clock noise.
    ``gpu_agreement_max_ratio``: the largest ratio, either way, between the
    one-call GPU time and the steady-state time per call that still
    corroborates the Day 74 rate.
    ``roof_excess_tolerance``: how far a rate may sit above a roof before the
    roof, or its traffic model, cannot hold.
    """

    intensity_model: str
    dispatch_overhead_share_min: float
    near_roof_share_min: float
    timing_tolerance: float
    gpu_agreement_max_ratio: float
    roof_excess_tolerance: float
    require_fma_scales_with_work: bool

    def __post_init__(self) -> None:
        if self.intensity_model not in TRAFFIC_MODELS:
            raise MatmulBottleneckError(
                f"intensity_model must be one of {', '.join(TRAFFIC_MODELS)}"
            )
        require_fraction(
            self.dispatch_overhead_share_min,
            "dispatch_overhead_share_min",
            0.0,
            1.0,
            open_low=True,
        )
        require_fraction(
            self.near_roof_share_min, "near_roof_share_min", 0.0, 1.0, open_low=True
        )
        require_fraction(self.timing_tolerance, "timing_tolerance", 0.0, 1.0)
        require_fraction(
            self.gpu_agreement_max_ratio,
            "gpu_agreement_max_ratio",
            1.0,
            MAX_AGREEMENT_RATIO,
        )
        require_fraction(self.roof_excess_tolerance, "roof_excess_tolerance", 0.0, 1.0)
        if type(self.require_fma_scales_with_work) is not bool:
            raise MatmulBottleneckError(
                "require_fma_scales_with_work must be a boolean"
            )

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in sorted(_RULE_FIELDS)}


@dataclass(frozen=True)
class MappingRule:
    """How much slower one mapping must run to count as worse.

    ``slower_ratio_min`` divides the median rate of x over columns by that of
    x over rows; a smaller ratio is within noise and claims nothing.
    """

    slower_ratio_min: float

    def __post_init__(self) -> None:
        require_fraction(
            self.slower_ratio_min, "slower_ratio_min", 1.0, MAX_AGREEMENT_RATIO
        )

    def to_mapping(self) -> dict[str, object]:
        return {"slower_ratio_min": self.slower_ratio_min}


@dataclass(frozen=True)
class BottleneckConfig:
    """One named, versioned set of bottleneck thresholds."""

    name: str
    rule: BottleneckRule
    mapping: MappingRule

    def __post_init__(self) -> None:
        if type(self.name) is not str or _NAME.fullmatch(self.name) is None:
            raise MatmulBottleneckError("config name must be a lowercase slug")
        if type(self.rule) is not BottleneckRule:
            raise TypeError("rule must be a BottleneckRule")
        if type(self.mapping) is not MappingRule:
            raise TypeError("mapping must be a MappingRule")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping that a report embeds and its digest covers."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "rule": self.rule.to_mapping(),
            "mapping": self.mapping.to_mapping(),
        }

    def fingerprint(self) -> str:
        """Bind every threshold to one SHA-256 digest of its canonical JSON."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _config(payload: object) -> BottleneckConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise MatmulBottleneckError(
            f"config schema_version must be {CONFIG_SCHEMA_VERSION}"
        )
    rule = _fields(config["rule"], _RULE_FIELDS, "rule")
    mapping = _fields(config["mapping"], _MAPPING_FIELDS, "mapping")
    return BottleneckConfig(
        name=config["name"],
        rule=BottleneckRule(**rule),
        mapping=MappingRule(**mapping),
    )


def config_from_mapping(payload: object) -> BottleneckConfig:
    """Parse thresholds, reporting every contract violation as one error type."""

    try:
        return _config(payload)
    except TypeError as error:
        raise MatmulBottleneckError(f"invalid bottleneck config: {error}") from error


def load_config(path: Path) -> BottleneckConfig:
    """Read one bounded, duplicate-free JSON config file and validate it."""

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise MatmulBottleneckError(str(error)) from error
    return config_from_mapping(payload)
