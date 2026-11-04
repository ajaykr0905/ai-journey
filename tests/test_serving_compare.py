from fractions import Fraction

import pytest

from ai_journey.serving_compare import (
    ComparisonKey,
    ServingComparisonError,
    ServingSnapshot,
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
