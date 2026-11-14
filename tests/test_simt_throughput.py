from fractions import Fraction

import pytest

from ai_journey.resource_accounting import ResourceAccountingError
from ai_journey.roofline import KernelWorkload, matmul_workload
from ai_journey.simt_device import PINNED_DEVICES, SimtError
from ai_journey.simt_throughput import (
    device_roofline,
    device_throughput,
    rgb_to_grayscale_workload,
    roofline_on_device,
    vector_add_workload,
)

A100 = PINNED_DEVICES["a100-sxm4-80gb"]
V100 = PINNED_DEVICES["tesla-v100-sxm2"]
T4 = PINNED_DEVICES["tesla-t4"]


def test_ridge_points_divide_derived_fp32_peaks_by_declared_bandwidth():
    # A100: 19,491.84 GFLOP/s / 2,039 GB/s = 9.5595 FLOPs per byte.
    # V100: 15,667.2 / 900 = 17.408. T4: 8,140.8 / 300 = 27.136.
    expected = {
        A100: Fraction(19_491_840, 2_039_000),
        V100: Fraction(15_667_200, 900_000),
        T4: Fraction(8_140_800, 300_000),
    }
    for device, ridge in expected.items():
        throughput = device_throughput(device)
        hardware = device_roofline(device)
        assert throughput.ridge_point_flops_per_byte == ridge
        assert hardware.ridge_point == float(ridge)
        assert hardware.peak_flops_per_second == device.peak_fp32_flops_per_second
        assert throughput.to_mapping()["ridge_point_flops_per_byte"] == float(ridge)
    assert float(expected[V100]) == 17.408
    assert float(expected[T4]) == 27.136


def test_vector_add_is_memory_bound_on_every_declared_device():
    # 2^24 float32 adds: 2^24 FLOPs over 3 * 4 * 2^24 bytes = 1/12 FLOP/byte.
    workload = vector_add_workload(elements=2**24, dtype="float32")
    assert workload == KernelWorkload(
        kind="vector_add",
        flops=16_777_216,
        bytes_read=134_217_728,
        bytes_written=67_108_864,
    )
    assert workload.arithmetic_intensity == 1 / 12
    for device in (A100, V100, T4):
        estimate = roofline_on_device(workload, device)
        assert estimate.bottleneck == "memory"
        assert estimate.lower_bound_seconds == 201_326_592 / (
            device.dram_bandwidth_bytes_per_second
        )
        assert estimate.attainable_flops_per_second == pytest.approx(
            device.dram_bandwidth_bytes_per_second / 12
        )


def test_grayscale_counts_declared_bytes_and_flops_per_pixel():
    # 1920 x 1080 = 2,073,600 pixels; 3 bytes in, 1 byte out, and 5 FLOPs
    # (three weighted channel products plus two adds) per pixel.
    workload = rgb_to_grayscale_workload(
        width=1920,
        height=1080,
        input_bytes_per_pixel=3,
        output_bytes_per_pixel=1,
        flops_per_pixel=5,
    )

    assert (workload.flops, workload.bytes_read, workload.bytes_written) == (
        10_368_000,
        6_220_800,
        2_073_600,
    )
    assert workload.arithmetic_intensity == 1.25
    assert roofline_on_device(workload, T4).bottleneck == "memory"


def test_dense_matmul_from_the_roofline_lab_is_compute_bound_here():
    # 4096^3 float32 matmul: 2 * 4096^3 FLOPs / (3 * 4096^2 * 4 bytes) = 682.7
    # FLOPs per byte, far above every declared ridge point.
    workload = matmul_workload(rows=4096, inner=4096, columns=4096, dtype="float32")
    for device in (A100, V100, T4):
        estimate = roofline_on_device(workload, device)
        assert estimate.bottleneck == "compute"
        assert estimate.attainable_flops_per_second == device.peak_fp32_flops_per_second


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: device_throughput("t4"), TypeError, "SimtDevice"),
        (lambda: device_roofline(None), TypeError, "SimtDevice"),
        (
            lambda: vector_add_workload(elements=0, dtype="float32"),
            SimtError,
            "elements",
        ),
        (
            lambda: vector_add_workload(elements=8, dtype="float8"),
            ResourceAccountingError,
            "unsupported dtype",
        ),
        (
            lambda: rgb_to_grayscale_workload(
                width=4,
                height=4,
                input_bytes_per_pixel=3,
                output_bytes_per_pixel=1,
                flops_per_pixel=False,
            ),
            TypeError,
            "flops_per_pixel",
        ),
    ],
)
def test_throughput_inputs_are_strict(call, error, message):
    with pytest.raises(error, match=message):
        call()
