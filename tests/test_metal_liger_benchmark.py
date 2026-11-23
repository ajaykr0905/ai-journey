import numpy
import pytest

from ai_journey.kernel_metal_benchmark import DeviceTiming
from ai_journey.metal_fused_benchmark import side_rates
from ai_journey.metal_liger_benchmark import (
    TRAFFIC_MODEL,
    benchmark_point,
    check_outputs,
    modeled_bytes,
)
from ai_journey.metal_liger_config import LigerShape, compulsory_bytes
from ai_journey.metal_liger_plan import TORCH_SIDES
from ai_journey.metal_liger_reference import dense_layout, liger_inputs, liger_reference
from ai_journey.metal_liger_source import (
    CROSS_ENTROPY,
    FAMILIES,
    FAMILY_KERNELS,
    RMSNORM,
    SWIGLU,
    MetalLigerError,
)
from ai_journey.metal_liger_speedup import speedup, vendor_label
from test_kernel_metal_runtime import requires_metal
from test_metal_liger_emulator import LigerEmulator, ticking_clock
from test_metal_liger_runtime import liger_runtime


def test_the_traffic_model_gives_each_side_its_declared_bytes():
    layout = dense_layout(8, 1000)
    n, r = 8000, 8

    assert modeled_bytes("liger_rmsnorm", RMSNORM, layout) == 12 * n + 8 * r
    assert modeled_bytes("torch_rmsnorm", RMSNORM, layout) == 40 * n + 20 * r
    assert modeled_bytes("liger_swiglu", SWIGLU, layout) == 12 * n + 8 * r
    assert modeled_bytes("torch_swiglu", SWIGLU, layout) == 20 * n
    assert modeled_bytes("liger_cross_entropy", CROSS_ENTROPY, layout) == 4 * n + 16 * r
    assert modeled_bytes("torch_cross_entropy", CROSS_ENTROPY, layout) == 8 * n + 16 * r
    assert set(TRAFFIC_MODEL) == set(FAMILIES)
    with pytest.raises(MetalLigerError, match="not a side"):
        modeled_bytes("torch_swiglu", RMSNORM, layout)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_point_times_both_sides_in_rotation_on_one_batch(family):
    backend = LigerEmulator()
    shape = LigerShape(family, 16, 300)
    point = benchmark_point(
        backend,
        seed=80,
        index=0,
        shape=shape,
        inputs="normal",
        warmup=1,
        repeats=4,
        target_bytes_per_sample=1 << 16,
        clock=ticking_clock(),
    ).to_mapping()

    kernel, torch = FAMILY_KERNELS[family], TORCH_SIDES[family]
    assert point["side_order"] == [kernel, torch]
    layout = shape.layout()
    compulsory = compulsory_bytes(family, layout)
    assert point["compulsory_bytes_per_call"] == compulsory
    calls = -(-(1 << 16) // compulsory)
    for name in (kernel, torch):
        side = point["sides"][name]
        assert side["calls_per_sample"] == calls
        assert len(side["samples_ns"]) == 4
        assert side["modeled_bytes_per_call"] == modeled_bytes(name, family, layout)
        assert point["checks"][name]["max_bound_ratio"] <= 1
    expected = 4 * layout.elements if family == CROSS_ENTROPY else None
    assert point["unwritten_softmax_bytes"] == expected
    # Each warmup and timed sample of each side synchronizes twice.
    assert backend.synchronized == 2 * (2 * 1 + 2 * 4)


def test_an_output_beyond_its_bound_stops_the_run():
    inputs = liger_inputs(80, SWIGLU, 1, 0, dense_layout(4, 64), "normal")
    reference = liger_reference(inputs)
    good = reference.output.astype(numpy.float32)
    outputs = {"liger_swiglu": good, "torch_swiglu": good}
    checks = check_outputs(outputs, SWIGLU, reference, good.size, "point")
    assert checks["liger_swiglu"].max_bound_ratio <= 0.5

    bad = good.copy()
    bad[3] += numpy.float32(0.01)
    with pytest.raises(MetalLigerError, match="torch_swiglu exceeds its bound"):
        check_outputs({**outputs, "torch_swiglu": bad}, SWIGLU, reference, 256, "p")
    with pytest.raises(MetalLigerError, match="no packed float32"):
        check_outputs({"liger_swiglu": good[:-1]}, SWIGLU, reference, 256, "p")


def _point(family, kernel_samples, torch_samples):
    layout = dense_layout(4, 256)
    compulsory = compulsory_bytes(family, layout)
    names = (FAMILY_KERNELS[family], TORCH_SIDES[family])
    return {
        "family": family,
        "index": 0,
        "shape": {"rows": 4, "cols": 256},
        "sides": {
            name: side_rates(
                DeviceTiming(name, 1, 0, len(samples), tuple(samples), "injected"),
                compulsory,
                modeled_bytes(name, family, layout),
            )
            for name, samples in zip(names, (kernel_samples, torch_samples))
        },
    }


def test_the_speedup_applies_the_declared_1_1_rule_and_the_predictions():
    fast = _point(RMSNORM, [100, 101, 102, 103], [200, 201, 202, 203])
    even = _point(CROSS_ENTROPY, [100, 140, 101, 139], [105, 106, 107, 108])
    slow = _point(SWIGLU, [300, 301, 302, 303], [100, 101, 102, 103])
    gains = speedup([fast, slow, even], 100.0, 1.1, "2.15.0")

    rows = gains["points"]
    assert rows[0]["to_torch"]["verdict"] == "faster" and rows[0]["held"] is True
    assert rows[1]["to_torch"]["verdict"] == "slower" and rows[1]["held"] is False
    assert rows[2]["to_torch"]["verdict"] == "no_clear_difference"
    assert rows[2]["held"] is True
    assert rows[0]["to_torch"]["ratio"] == 201 / 101
    kernel = rows[0]["kernel"]
    assert kernel["effective_fraction_of_ceiling"] == (
        kernel["effective_gb_per_second"] / 100.0
    )
    assert [record["missed_at"] for record in gains["hypotheses"]] == [[], [0], []]
    assert gains["vendors"][RMSNORM] == vendor_label(RMSNORM, "2.15.0")
    assert "PyTorch 2.15.0" in gains["vendors"][CROSS_ENTROPY]


@requires_metal
@pytest.mark.parametrize("family", FAMILIES)
def test_a_small_point_runs_both_sides_on_the_apple_gpu(family):
    point = benchmark_point(
        liger_runtime(),
        seed=80,
        index=0,
        shape=LigerShape(family, 64, 2049),
        inputs="normal",
        warmup=1,
        repeats=4,
        target_bytes_per_sample=1 << 20,
    ).to_mapping()

    assert all(side["timing"]["median_ns"] > 0 for side in point["sides"].values())
    assert all(check["max_bound_ratio"] <= 1 for check in point["checks"].values())
