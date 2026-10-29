from copy import deepcopy

import pytest

from ai_journey.resource_accounting import DecoderSpec, ResourceAccountingError
from ai_journey.roofline_report import (
    RooflinePlan,
    build_roofline_report,
    seal_roofline_report,
    verify_roofline_report,
)


def make_plan() -> RooflinePlan:
    return RooflinePlan(
        model=DecoderSpec(
            vocab_size=100,
            layer_count=2,
            hidden_size=16,
            attention_heads=4,
            kv_heads=2,
            feed_forward_size=32,
        ),
        batch_size=3,
        sequence_length=5,
        dtype="bfloat16",
        peak_flops_per_second=20,
        memory_bandwidth_bytes_per_second=10,
    )


def test_roofline_report_reconciles_operations_and_summary():
    report = build_roofline_report(make_plan())

    assert report["schema_version"] == 1
    assert report["kind"] == "analytical_decoder_roofline"
    assert report["measured"] is False
    assert report["hardware"]["ridge_point_flops_per_byte"] == 2
    assert report["summary"]["operation_count"] == 10
    assert report["summary"]["bottleneck_counts"] == {
        "compute": 6,
        "memory": 4,
        "ridge": 0,
    }
    assert report["summary"]["total_flops"] == sum(
        item["flops"] for item in report["operations"].values()
    )
    assert report["operations"]["block_layer_norms"]["bottleneck"] == "memory"
    assert report["operations"]["attention_query"]["bottleneck"] == "compute"


def test_roofline_report_seal_is_stable_without_mutating_input():
    report = build_roofline_report(make_plan())
    first = seal_roofline_report(report)
    second = seal_roofline_report(report)

    assert first == second
    assert "integrity" not in report
    assert len(first["integrity"]["digest"]) == 64


def test_roofline_report_verification_recomputes_every_field():
    sealed = seal_roofline_report(build_roofline_report(make_plan()))

    assert verify_roofline_report(sealed) is sealed
    changed = deepcopy(sealed)
    changed["operations"]["attention_query"]["flops"] += 1
    changed = seal_roofline_report(changed)
    with pytest.raises(ResourceAccountingError, match="does not match recomputation"):
        verify_roofline_report(changed)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("schema_version", True, "unsupported roofline report schema"),
        ("measured", 0, "must remain unmeasured"),
        ("kind", "measured_roofline", "unexpected roofline report kind"),
    ],
)
def test_roofline_report_rejects_serially_distinct_contract_values(
    field, value, message
):
    report = seal_roofline_report(build_roofline_report(make_plan()))
    report[field] = value
    report = seal_roofline_report(report)

    with pytest.raises(ResourceAccountingError, match=message):
        verify_roofline_report(report)
