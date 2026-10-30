import pytest

from ai_journey.kernel_fusion import ElementwiseStage, compare_elementwise_fusion
from ai_journey.resource_accounting import ResourceAccountingError
from ai_journey.roofline import RooflineHardware


def hardware():
    return RooflineHardware(
        peak_flops_per_second=1_000_000,
        memory_bandwidth_bytes_per_second=1_000,
    )


def test_fusion_eliminates_only_materialized_intermediate_traffic():
    comparison = compare_elementwise_fusion(
        elements=1_024,
        dtype="bfloat16",
        stages=(
            ElementwiseStage("bias", 1, auxiliary_reads_per_element=1),
            ElementwiseStage("gelu", 8),
            ElementwiseStage("residual", 1, auxiliary_reads_per_element=1),
        ),
        hardware=hardware(),
    )

    assert comparison.unfused.workload.flops == 10_240
    assert comparison.fused.workload.flops == 10_240
    assert comparison.unfused.workload.bytes_read == 10_240
    assert comparison.unfused.workload.bytes_written == 6_144
    assert comparison.fused.workload.bytes_read == 6_144
    assert comparison.fused.workload.bytes_written == 2_048
    assert comparison.intermediate_bytes_eliminated == 8_192
    assert comparison.traffic_reduction_fraction == 0.5
    assert comparison.roofline_lower_bound_ratio == 2
    assert comparison.unfused.bottleneck == comparison.fused.bottleneck == "memory"


def test_single_stage_has_no_intermediate_traffic_to_eliminate():
    comparison = compare_elementwise_fusion(
        elements=8,
        dtype="float32",
        stages=(ElementwiseStage("relu", 1),),
        hardware=hardware(),
    )

    assert comparison.intermediate_bytes_eliminated == 0
    assert comparison.traffic_reduction_fraction == 0
    assert comparison.roofline_lower_bound_ratio == 1


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: ElementwiseStage("", 1),
            "stage name must be nonempty",
        ),
        (
            lambda: ElementwiseStage("bias", True),
            "flops_per_element must be an integer",
        ),
        (
            lambda: ElementwiseStage("bias", 1, -1),
            "auxiliary_reads_per_element must be nonnegative",
        ),
        (
            lambda: compare_elementwise_fusion(
                elements=1,
                dtype="float32",
                stages=(),
                hardware=hardware(),
            ),
            "stages must be a nonempty tuple",
        ),
        (
            lambda: compare_elementwise_fusion(
                elements=1,
                dtype="float32",
                stages=(ElementwiseStage("same", 1), ElementwiseStage("same", 2)),
                hardware=hardware(),
            ),
            "stage names must be unique",
        ),
    ],
)
def test_fusion_contract_rejects_ambiguous_inputs(call, message):
    with pytest.raises((TypeError, ResourceAccountingError), match=message):
        call()
