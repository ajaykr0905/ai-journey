import pytest

from ai_journey.serving_compare import ComparisonKey, ServingComparisonError


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
