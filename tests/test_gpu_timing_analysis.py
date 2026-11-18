import pytest

from ai_journey.gpu_timing_analysis import INFERENCE_BASIS, analyze, fit_sides
from ai_journey.gpu_timing_clocks import (
    CLOCKS,
    DISPATCH,
    EVENT,
    SYNCHRONIZED,
    GpuTimingError,
)
from ai_journey.gpu_timing_fit import FIXED_DOMINANT, SIZE_DOMINANT
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.matmul_metal_runtime import SIDES
from test_gpu_timing_sweep import FIXED_NS, NS_PER_FLOP, StandInBackend, sweep

SHAPES = (
    matmul_shape(1, 1, 1),
    matmul_shape(16, 16, 16),
    matmul_shape(64, 64, 64),
    matmul_shape(256, 128, 256),
)
CEILING = 4618.0


def measured():
    return sweep(StandInBackend(), shapes=SHAPES)


def test_the_lines_recover_the_virtual_costs_on_each_clock():
    _, points = measured()

    fits = fit_sides(points)

    for side in SIDES:
        event = fits[side][EVENT]
        assert event.intercept_ns == pytest.approx(3_000 + FIXED_NS, abs=1)
        assert event.slope_ns_per_flop == pytest.approx(NS_PER_FLOP[side], rel=1e-4)
        synchronized = fits[side][SYNCHRONIZED]
        assert synchronized.intercept_ns == pytest.approx(48_000, abs=1)
        dispatch = fits[side][DISPATCH]
        assert dispatch.intercept_ns == pytest.approx(3_000)
        assert dispatch.slope_ns_per_flop == pytest.approx(0, abs=1e-12)


def test_the_overhead_estimate_quotes_the_empty_kernel_and_the_smallest_shape():
    empty, points = measured()

    analysis = analyze(empty, points, CEILING)

    estimate = analysis["overhead_estimate"]
    assert estimate["empty_kernel_median_ns"] == {
        EVENT: 3_000 + FIXED_NS,
        SYNCHRONIZED: 48_000,
        DISPATCH: 3_000,
    }
    assert estimate["empty_kernel_dispatch_to_synchronized_ratio"] == 3_000 / 48_000
    assert estimate["empty_kernel_host_minus_event"]["value_ns"] == 40_000
    assert estimate["empty_kernel_host_minus_event"]["cross_domain"] is True
    smallest = estimate["smallest_shape"]
    assert (smallest["m"], smallest["n"], smallest["k"]) == (1, 1, 1)
    assert set(smallest["median_ns"]) == set(SIDES)
    ratio = analysis["fits"]["matmul_x_rows"][EVENT]["intercept_to_empty_kernel_ratio"]
    assert ratio == pytest.approx(1.0, abs=1e-3)


def test_small_points_are_fixed_dominant_and_large_points_are_not():
    empty, points = measured()

    analysis = analyze(empty, points, CEILING)

    first, last = analysis["points"][0], analysis["points"][-1]
    for side in SIDES:
        assert first["sides"][side]["clocks"][EVENT]["regime"] == FIXED_DOMINANT
        assert last["sides"][side]["clocks"][EVENT]["regime"] == SIZE_DOMINANT
        entry = first["sides"][side]["clocks"][EVENT]
        assert entry["fixed_ns"] + entry["size_ns"] == entry["median_ns"]
        assert entry["iqr_ns"] == 0
        # Dispatch does not grow with work, so it stays fixed-dominant.
        assert last["sides"][side]["clocks"][DISPATCH]["regime"] == FIXED_DOMINANT


def test_rates_use_the_event_median_and_the_day_74_ceiling():
    empty, points = measured()

    analysis = analyze(empty, points, CEILING)

    last = analysis["points"][-1]
    flops = SHAPES[-1].flops
    for side in SIDES:
        entry = last["sides"][side]
        median = points[-1].timing(side).median_ns(EVENT)
        assert entry["event_gflops_per_second"] == flops / median
        assert entry["event_fraction_of_fma_ceiling"] == flops / median / CEILING
        assert entry["host_minus_event"] == {
            "value_ns": 40_000,
            "cross_domain": True,
            "minuend": SYNCHRONIZED,
            "subtrahend": EVENT,
        }
    assert "no hardware counter" in analysis["inference_basis"] == INFERENCE_BASIS
    assert analysis["fma_ceiling_gflops_per_second"] == CEILING
    assert set(analysis["fits"]["torch_matmul"]) == set(CLOCKS)


def test_bad_inputs_are_refused():
    empty, points = measured()

    with pytest.raises(TypeError):
        analyze(points[0], points, CEILING)
    with pytest.raises(TypeError):
        analyze(empty, list(points), CEILING)
    with pytest.raises(GpuTimingError):
        analyze(empty, points, 4618)
    with pytest.raises(GpuTimingError, match="at least 3"):
        analyze(empty, points[:2], CEILING)
