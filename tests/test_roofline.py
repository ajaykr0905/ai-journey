import pytest

from ai_journey.resource_accounting import ResourceAccountingError
from ai_journey.roofline import (
    KernelWorkload,
    RooflineHardware,
    analyze_roofline,
    layer_norm_workload,
    matmul_workload,
)


def test_matmul_counts_flops_and_compulsory_tensor_traffic():
    workload = matmul_workload(
        rows=4,
        inner=8,
        columns=16,
        batch_count=2,
        dtype="bfloat16",
    )

    assert workload == KernelWorkload(
        kind="matmul",
        flops=2_048,
        bytes_read=640,
        bytes_written=256,
    )
    assert workload.total_bytes == 896
    assert workload.arithmetic_intensity == pytest.approx(2.2857142857142856)


def test_layer_norm_exposes_operation_and_input_pass_assumptions():
    baseline = layer_norm_workload(
        tokens=4,
        hidden_size=8,
        dtype="bfloat16",
    )
    one_pass = layer_norm_workload(
        tokens=4,
        hidden_size=8,
        dtype="bfloat16",
        operations_per_element=6.5,
        input_passes=1,
    )

    assert baseline == KernelWorkload(
        kind="layer_norm",
        flops=256,
        bytes_read=128,
        bytes_written=64,
    )
    assert baseline.arithmetic_intensity == pytest.approx(4 / 3)
    assert one_pass.flops == 208
    assert one_pass.total_bytes == 128


def test_roofline_classifies_compute_and_memory_limits_from_declared_peaks():
    hardware = RooflineHardware(
        peak_flops_per_second=100,
        memory_bandwidth_bytes_per_second=10,
    )
    compute_bound = analyze_roofline(
        KernelWorkload("matmul", flops=200, bytes_read=8, bytes_written=2),
        hardware,
    )
    memory_bound = analyze_roofline(
        KernelWorkload("layer_norm", flops=20, bytes_read=18, bytes_written=2),
        hardware,
    )

    assert hardware.ridge_point == 10
    assert compute_bound.bottleneck == "compute"
    assert compute_bound.compute_seconds == 2
    assert compute_bound.memory_seconds == 1
    assert compute_bound.lower_bound_seconds == 2
    assert compute_bound.attainable_flops_per_second == 100
    assert memory_bound.bottleneck == "memory"
    assert memory_bound.lower_bound_seconds == 2
    assert memory_bound.attainable_flops_per_second == 10


def test_roofline_preserves_exact_ridge_and_rejects_invalid_hardware():
    ridge = analyze_roofline(
        KernelWorkload("ridge", flops=100, bytes_read=5, bytes_written=5),
        RooflineHardware(
            peak_flops_per_second=100, memory_bandwidth_bytes_per_second=10
        ),
    )

    assert ridge.bottleneck == "ridge"
    assert ridge.lower_bound_seconds == 1
    with pytest.raises(ResourceAccountingError, match="positive and finite"):
        RooflineHardware(
            peak_flops_per_second=float("inf"),
            memory_bandwidth_bytes_per_second=10,
        )
    with pytest.raises(TypeError, match="KernelWorkload"):
        analyze_roofline("not-workload", ridge.hardware)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: matmul_workload(rows=True, inner=8, columns=16, dtype="bfloat16"),
            "rows must be an integer",
        ),
        (
            lambda: matmul_workload(rows=4, inner=0, columns=16, dtype="bfloat16"),
            "inner must be positive",
        ),
        (
            lambda: layer_norm_workload(
                tokens=4,
                hidden_size=8,
                dtype="bfloat16",
                operations_per_element=float("nan"),
            ),
            "operations_per_element must be positive and finite",
        ),
        (
            lambda: layer_norm_workload(tokens=4, hidden_size=8, dtype="float24"),
            "unsupported dtype",
        ),
    ],
)
def test_kernel_workloads_reject_ambiguous_or_invalid_inputs(call, message):
    with pytest.raises((TypeError, ResourceAccountingError), match=message):
        call()
