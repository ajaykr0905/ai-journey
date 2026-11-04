"""Exact regression decisions for comparable serving evidence."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


class ServingComparisonError(ValueError):
    """Raised when serving evidence cannot be compared safely."""


def _require_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ServingComparisonError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ComparisonKey:
    """Identity fields that must match across baseline and candidate runs."""

    protocol_version: int
    workload_sha256: str
    model_sha256: str
    trace_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.protocol_version, bool)
            or not isinstance(self.protocol_version, int)
            or self.protocol_version <= 0
        ):
            raise ServingComparisonError("protocol_version must be positive")
        for name in ("workload_sha256", "model_sha256", "trace_sha256"):
            _require_sha256(getattr(self, name), name)


@dataclass(frozen=True)
class ServingSnapshot:
    """Exact metrics and identity for one bounded serving run."""

    key: ComparisonKey
    engine_label: str
    p95_ttft_ns: int
    output_tokens_per_second: Fraction
    goodput: Fraction
    failure_rate: Fraction
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, ComparisonKey):
            raise TypeError("key must be ComparisonKey")
        if not isinstance(self.engine_label, str) or not self.engine_label.strip():
            raise ServingComparisonError("engine_label must be nonempty")
        for name in ("p95_ttft_ns", "peak_memory_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ServingComparisonError(f"{name} must be positive")
        if not isinstance(self.output_tokens_per_second, Fraction):
            raise TypeError("output_tokens_per_second must be Fraction")
        if self.output_tokens_per_second <= 0:
            raise ServingComparisonError("output_tokens_per_second must be positive")
        for name in ("goodput", "failure_rate"):
            value = getattr(self, name)
            if not isinstance(value, Fraction):
                raise TypeError(f"{name} must be Fraction")
            if not 0 <= value <= 1:
                raise ServingComparisonError(f"{name} must be in [0, 1]")
