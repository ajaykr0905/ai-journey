import pytest

from ai_journey.distributed_capacity import ClusterTopology, LinkProfile
from ai_journey.resource_accounting import DecoderSpec
from ai_journey.zero_report import (
    ZeroComparisonPlan,
    ZeroReportError,
    build_zero_comparison_report,
    seal_zero_comparison_report,
    verify_zero_comparison_report,
)


def _plan() -> ZeroComparisonPlan:
    return ZeroComparisonPlan(
        model=DecoderSpec(32, 2, 16, 4, 2, 32),
        topology=ClusterTopology(
            world_size=4,
            workers_per_node=2,
            intra_node=LinkProfile(1_000_000, 0.00001),
            inter_node=LinkProfile(100_000, 0.001),
        ),
        parameter_dtype="bfloat16",
        gradient_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=2,
        master_parameter_dtype="float32",
        activation_bytes=1_000,
        gradient_bucket_bytes=4_096,
    )


def test_zero_report_compares_all_stages_from_one_declared_plan():
    report = build_zero_comparison_report(_plan())

    assert report["kind"] == "analytical_zero_stage_comparison"
    assert report["measured"] is False
    assert [stage["stage"] for stage in report["stages"]] == [0, 1, 2, 3]
    assert [len(stage["collectives"]) for stage in report["stages"]] == [1, 2, 2, 2]
    assert [stage["max_peak_bytes"] for stage in report["stages"]] == [
        103_816,
        52_408,
        39_368,
        35_272,
    ]
    assert all(stage["wire_bytes_per_rank"] > 0 for stage in report["stages"])
    assert report["limitations"][0].startswith("Analytical")


def test_zero_report_is_deterministic_and_rebuildable():
    first = build_zero_comparison_report(_plan())
    second = build_zero_comparison_report(_plan())
    assert first == second

    sealed = seal_zero_comparison_report(first)
    assert verify_zero_comparison_report(sealed) is sealed


def test_zero_report_rejects_resealed_derived_field_drift():
    sealed = seal_zero_comparison_report(build_zero_comparison_report(_plan()))
    drifted = {
        **sealed,
        "stages": [
            {**sealed["stages"][0], "max_peak_bytes": 1},
            *sealed["stages"][1:],
        ],
    }
    drifted = seal_zero_comparison_report(drifted)
    with pytest.raises(ZeroReportError, match="derived fields"):
        verify_zero_comparison_report(drifted)
