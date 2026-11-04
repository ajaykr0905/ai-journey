from fractions import Fraction

import pytest

from ai_journey.serving_compare import (
    ComparisonKey,
    RegressionThresholds,
    ServingComparisonError,
    ServingSnapshot,
    goodput_within_threshold,
    throughput_within_threshold,
    ttft_within_threshold,
)


def _digest(character: str) -> str:
    return character * 64


def test_comparison_key_binds_protocol_workload_model_and_trace():
    key = ComparisonKey(1, _digest("a"), _digest("b"), _digest("c"))

    assert key.protocol_version == 1
    assert key.trace_sha256 == _digest("c")


@pytest.mark.parametrize("protocol_version", [True, 0, -1])
def test_comparison_key_requires_positive_protocol_version(protocol_version):
    with pytest.raises(ServingComparisonError, match="protocol_version"):
        ComparisonKey(protocol_version, _digest("a"), _digest("b"), _digest("c"))


def test_comparison_key_rejects_noncanonical_digests():
    with pytest.raises(ServingComparisonError, match="workload_sha256"):
        ComparisonKey(1, "A" * 64, _digest("b"), _digest("c"))


def _key() -> ComparisonKey:
    return ComparisonKey(1, _digest("a"), _digest("b"), _digest("c"))


def _snapshot(label="engine", **overrides) -> ServingSnapshot:
    values = {
        "key": _key(),
        "engine_label": label,
        "p95_ttft_ns": 100,
        "output_tokens_per_second": Fraction(100),
        "goodput": Fraction(9, 10),
        "failure_rate": Fraction(1, 100),
        "peak_memory_bytes": 100,
    }
    values.update(overrides)
    return ServingSnapshot(**values)


def test_snapshot_preserves_exact_serving_metrics():
    snapshot = ServingSnapshot(
        _key(), "baseline", 100, Fraction(400, 3), Fraction(9, 10), Fraction(1, 10), 64
    )

    assert snapshot.output_tokens_per_second == Fraction(400, 3)
    assert snapshot.goodput == Fraction(9, 10)


@pytest.mark.parametrize(
    "goodput,failure_rate",
    [(Fraction(-1, 10), Fraction(0)), (Fraction(1), Fraction(2))],
)
def test_snapshot_rejects_unbounded_rates(goodput, failure_rate):
    with pytest.raises(ServingComparisonError, match=r"in \[0, 1\]"):
        ServingSnapshot(
            _key(), "candidate", 100, Fraction(10), goodput, failure_rate, 64
        )


def test_regression_thresholds_require_exact_bounded_fractions():
    thresholds = RegressionThresholds(
        Fraction(1, 10),
        Fraction(1, 20),
        Fraction(1, 100),
        Fraction(1, 100),
        Fraction(1, 10),
    )

    assert thresholds.max_throughput_drop == Fraction(1, 20)

    with pytest.raises(ServingComparisonError, match="max_memory_increase"):
        RegressionThresholds(
            Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(2)
        )


def _thresholds(**overrides) -> RegressionThresholds:
    values = {
        "max_ttft_increase": Fraction(1, 10),
        "max_throughput_drop": Fraction(1, 10),
        "max_goodput_drop": Fraction(1, 100),
        "max_failure_increase": Fraction(1, 100),
        "max_memory_increase": Fraction(1, 10),
    }
    values.update(overrides)
    return RegressionThresholds(**values)


def test_ttft_comparison_uses_exact_relative_limit():
    assert ttft_within_threshold(_snapshot(), _snapshot(p95_ttft_ns=110), _thresholds())
    assert not ttft_within_threshold(
        _snapshot(), _snapshot(p95_ttft_ns=111), _thresholds()
    )


def test_ttft_comparison_rejects_mismatched_experiment_identity():
    other_key = ComparisonKey(2, _digest("a"), _digest("b"), _digest("c"))

    with pytest.raises(ServingComparisonError, match="different comparison keys"):
        ttft_within_threshold(_snapshot(), _snapshot(key=other_key), _thresholds())


def test_throughput_comparison_uses_exact_relative_floor():
    assert throughput_within_threshold(
        _snapshot(),
        _snapshot(output_tokens_per_second=Fraction(90)),
        _thresholds(),
    )
    assert not throughput_within_threshold(
        _snapshot(),
        _snapshot(output_tokens_per_second=Fraction(899, 10)),
        _thresholds(),
    )


def test_goodput_comparison_uses_absolute_percentage_point_drop():
    assert goodput_within_threshold(
        _snapshot(), _snapshot(goodput=Fraction(89, 100)), _thresholds()
    )
    assert not goodput_within_threshold(
        _snapshot(), _snapshot(goodput=Fraction(889, 1000)), _thresholds()
    )
