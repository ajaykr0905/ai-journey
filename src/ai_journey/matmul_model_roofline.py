"""A roofline at given ceilings, in exact rational arithmetic.

``roofline`` returns the attainable rate ``min(compute ceiling, intensity x
bandwidth ceiling)``, and ``Ceilings.ridge_point`` is ``compute ceiling /
bandwidth ceiling``, the intensity where the two limits meet. The units are
decimal: GFLOP/s and GB/s count 10**9 FLOPs and bytes per second, so FLOP per
byte times GB/s is GFLOP/s. A point exactly on the ridge is ``ridge``, as in
``roofline.analyze_roofline``, instead of either limit.

This module holds no hardware number. A caller passes ceilings that it
measured on its own host, or declared values for a model, under a label that
says which. Ceilings are decimal strings, so a report states them exactly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from .matmul_model_reference import MatmulModelError
from .matmul_model_traffic import ratio_mapping

BOUNDS = ("memory", "ridge", "compute")
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{0,5}[1-9])?")
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_CEILING_FIELDS = {"label", "compute_gflops", "bandwidth_gbps"}


def parse_decimal(text: object, name: str) -> Fraction:
    """Parse a positive decimal string with no exponent and no trailing zero."""

    if type(text) is not str or _DECIMAL.fullmatch(text) is None:
        raise MatmulModelError(
            f"{name} must be a canonical decimal string such as '19.5'"
        )
    value = Fraction(text)
    if value <= 0:
        raise MatmulModelError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class Ceilings:
    """A compute ceiling in GFLOP/s and a bandwidth ceiling in GB/s."""

    label: str
    compute_gflops: str
    bandwidth_gbps: str

    def __post_init__(self) -> None:
        if type(self.label) is not str or _LABEL.fullmatch(self.label) is None:
            raise MatmulModelError("ceiling label must be a lowercase slug")
        parse_decimal(self.compute_gflops, "compute_gflops")
        parse_decimal(self.bandwidth_gbps, "bandwidth_gbps")

    @property
    def compute(self) -> Fraction:
        return Fraction(self.compute_gflops)

    @property
    def bandwidth(self) -> Fraction:
        return Fraction(self.bandwidth_gbps)

    @property
    def ridge_point(self) -> Fraction:
        """Return the intensity in FLOP per byte where the two ceilings meet."""

        return self.compute / self.bandwidth

    def to_mapping(self) -> dict[str, str]:
        return {
            "label": self.label,
            "compute_gflops": self.compute_gflops,
            "bandwidth_gbps": self.bandwidth_gbps,
        }

    @classmethod
    def from_mapping(cls, mapping: object) -> Ceilings:
        """Parse ``{"label", "compute_gflops", "bandwidth_gbps"}`` exactly."""

        if not isinstance(mapping, Mapping) or set(mapping) != _CEILING_FIELDS:
            raise MatmulModelError(
                "ceilings must hold exactly label, compute_gflops and bandwidth_gbps"
            )
        return cls(
            mapping["label"], mapping["compute_gflops"], mapping["bandwidth_gbps"]
        )


@dataclass(frozen=True)
class RooflinePoint:
    """The attainable rate of one arithmetic intensity under one ceiling pair."""

    intensity: Fraction
    ceilings: Ceilings

    def __post_init__(self) -> None:
        if type(self.intensity) is not Fraction or self.intensity <= 0:
            raise MatmulModelError("intensity must be a positive Fraction")
        if type(self.ceilings) is not Ceilings:
            raise TypeError("ceilings must be Ceilings")

    @property
    def memory_gflops(self) -> Fraction:
        """Return the rate that the bandwidth ceiling allows at this intensity."""

        return self.intensity * self.ceilings.bandwidth

    @property
    def attainable_gflops(self) -> Fraction:
        return min(self.ceilings.compute, self.memory_gflops)

    @property
    def bound(self) -> str:
        if self.memory_gflops < self.ceilings.compute:
            return "memory"
        if self.memory_gflops > self.ceilings.compute:
            return "compute"
        return "ridge"

    @property
    def fraction_of_compute_ceiling(self) -> Fraction:
        return self.attainable_gflops / self.ceilings.compute

    def to_mapping(self) -> dict[str, object]:
        return {
            "intensity": ratio_mapping(self.intensity),
            "ridge_point": ratio_mapping(self.ceilings.ridge_point),
            "attainable_gflops": ratio_mapping(self.attainable_gflops),
            "bound": self.bound,
            "fraction_of_compute_ceiling": ratio_mapping(
                self.fraction_of_compute_ceiling
            ),
        }


def roofline(intensity: Fraction, ceilings: Ceilings) -> RooflinePoint:
    """Return the attainable GFLOP/s and the ridge of one intensity."""

    return RooflinePoint(intensity, ceilings)
