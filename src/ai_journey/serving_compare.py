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


@dataclass(frozen=True)
class RegressionThresholds:
    """Maximum exact degradation accepted for each serving metric."""

    max_ttft_increase: Fraction
    max_throughput_drop: Fraction
    max_goodput_drop: Fraction
    max_failure_increase: Fraction
    max_memory_increase: Fraction

    def __post_init__(self) -> None:
        for name in (
            "max_ttft_increase",
            "max_throughput_drop",
            "max_goodput_drop",
            "max_failure_increase",
            "max_memory_increase",
        ):
            value = getattr(self, name)
            if not isinstance(value, Fraction):
                raise TypeError(f"{name} must be Fraction")
            if not 0 <= value <= 1:
                raise ServingComparisonError(f"{name} must be in [0, 1]")


def _require_comparable(baseline: ServingSnapshot, candidate: ServingSnapshot) -> None:
    if not isinstance(baseline, ServingSnapshot) or not isinstance(
        candidate, ServingSnapshot
    ):
        raise TypeError("baseline and candidate must be ServingSnapshot")
    if baseline.key != candidate.key:
        raise ServingComparisonError("serving snapshots have different comparison keys")


def ttft_within_threshold(
    baseline: ServingSnapshot,
    candidate: ServingSnapshot,
    thresholds: RegressionThresholds,
) -> bool:
    """Return whether candidate p95 TTFT stays inside relative tolerance."""

    _require_comparable(baseline, candidate)
    if not isinstance(thresholds, RegressionThresholds):
        raise TypeError("thresholds must be RegressionThresholds")
    limit = Fraction(baseline.p95_ttft_ns) * (1 + thresholds.max_ttft_increase)
    return candidate.p95_ttft_ns <= limit


def throughput_within_threshold(
    baseline: ServingSnapshot,
    candidate: ServingSnapshot,
    thresholds: RegressionThresholds,
) -> bool:
    """Return whether candidate throughput stays above the relative floor."""

    _require_comparable(baseline, candidate)
    if not isinstance(thresholds, RegressionThresholds):
        raise TypeError("thresholds must be RegressionThresholds")
    floor = baseline.output_tokens_per_second * (1 - thresholds.max_throughput_drop)
    return candidate.output_tokens_per_second >= floor


def goodput_within_threshold(
    baseline: ServingSnapshot,
    candidate: ServingSnapshot,
    thresholds: RegressionThresholds,
) -> bool:
    """Return whether candidate goodput stays inside the absolute tolerance."""

    _require_comparable(baseline, candidate)
    if not isinstance(thresholds, RegressionThresholds):
        raise TypeError("thresholds must be RegressionThresholds")
    return candidate.goodput >= baseline.goodput - thresholds.max_goodput_drop


def failures_within_threshold(
    baseline: ServingSnapshot,
    candidate: ServingSnapshot,
    thresholds: RegressionThresholds,
) -> bool:
    """Return whether candidate failures stay inside the absolute tolerance."""

    _require_comparable(baseline, candidate)
    if not isinstance(thresholds, RegressionThresholds):
        raise TypeError("thresholds must be RegressionThresholds")
    return (
        candidate.failure_rate
        <= baseline.failure_rate + thresholds.max_failure_increase
    )


def memory_within_threshold(
    baseline: ServingSnapshot,
    candidate: ServingSnapshot,
    thresholds: RegressionThresholds,
) -> bool:
    """Return whether candidate peak memory stays inside relative tolerance."""

    _require_comparable(baseline, candidate)
    if not isinstance(thresholds, RegressionThresholds):
        raise TypeError("thresholds must be RegressionThresholds")
    limit = Fraction(baseline.peak_memory_bytes) * (1 + thresholds.max_memory_increase)
    return candidate.peak_memory_bytes <= limit
