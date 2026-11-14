import itertools
import math

import pytest

from ai_journey.memory_hierarchy_stream import (
    KERNEL_NAMES,
    MemoryHierarchyError,
    StreamBuffers,
)
from ai_journey.memory_hierarchy_sweep import (
    StreamMeasurement,
    curve_summary,
    measure_stream_sweep,
    stream_curves,
    validate_sweep,
)
from ai_journey.memory_hierarchy_timing import BatchTiming, RunBudget


def counting_clock(step=1_000):
    return itertools.count(0, step).__next__


def sweep(**changes):
    arguments = {
        "kernels": KERNEL_NAMES,
        "elements": (256, 1024),
        "dtype": "float64",
        "seed": 71,
        "warmup": 1,
        "repeats": 3,
        "target_bytes_per_sample": 8192,
        "clock": counting_clock(),
        **changes,
    }
    return measure_stream_sweep(**arguments)


def measurement(kernel, elements, samples, calls=1):
    timing = BatchTiming(calls, 0, len(samples), tuple(samples), "injected")
    return StreamMeasurement(kernel, elements, "float64", timing)


def test_every_kernel_is_measured_at_every_point_smallest_first():
    results = sweep()

    assert [(item.kernel, item.elements) for item in results] == [
        (kernel, elements) for elements in (256, 1024) for kernel in KERNEL_NAMES
    ]
    copy_small, triad_large = results[0].to_mapping(), results[-1].to_mapping()
    assert copy_small == {
        "kernel": "copy",
        "elements": 256,
        "working_set_bytes": 4096,
        "bytes_per_call": 4096,
        "flops_per_call": 0,
        "calls_per_sample": 2,
        "warmup": 1,
        "repeats": 3,
        "samples_ns": [1000, 1000, 1000],
        "timing": {
            "count": 3,
            "min_ns": 1000,
            "p25_ns": 1000,
            "median_ns": 1000,
            "p75_ns": 1000,
            "max_ns": 1000,
            "iqr_ns": 0,
        },
        "bytes_per_sample": 8192,
        "flops_per_sample": 0,
        "median_ns_per_call": 500.0,
        "median_gb_per_second": 8.192,
        "best_gb_per_second": 8.192,
        "median_gflops_per_second": 0.0,
    }
    assert triad_large["working_set_bytes"] == 3 * 1024 * 8
    assert triad_large["bytes_per_call"] == 5 * 1024 * 8
    assert triad_large["calls_per_sample"] == 1
    assert triad_large["flops_per_sample"] == 2048
    assert triad_large["median_gb_per_second"] == 40960 / 1000
    assert triad_large["median_gflops_per_second"] == 2.048


def test_a_kernel_that_skips_its_writes_fails_the_sweep(monkeypatch):
    real_bind = StreamBuffers.bind

    def lazy_bind(self, name, elements):
        if name == "add":
            return lambda: None
        return real_bind(self, name, elements)

    monkeypatch.setattr(StreamBuffers, "bind", lazy_bind)

    with pytest.raises(MemoryHierarchyError, match="add output"):
        sweep()


def test_the_run_budget_bounds_the_whole_sweep():
    clock = counting_clock(step=100_000_000)
    budget = RunBudget(1, clock)

    with pytest.raises(MemoryHierarchyError, match="max_runtime_seconds=1"):
        sweep(clock=clock, budget=budget)


def test_curves_follow_sweep_order_and_summaries_name_no_cache_level():
    results = (
        measurement("triad", 256, [100, 100, 100]),
        measurement("copy", 256, [100, 120, 140]),
        measurement("triad", 1024, [200, 210, 230]),
        measurement("triad", 4096, [1600, 1700, 1800]),
    )

    curves = stream_curves(results)

    assert list(curves) == ["triad", "copy"]
    assert [point["elements"] for point in curves["triad"]] == [256, 1024, 4096]
    assert curves["copy"] == [
        {
            "elements": 256,
            "working_set_bytes": 4096,
            "median_gb_per_second": 4096 / 120,
            "best_gb_per_second": 4096 / 100,
        }
    ]
    summary = curve_summary(curves["triad"])
    assert summary == {
        "peak_working_set_bytes": 3 * 1024 * 8,
        "peak_median_gb_per_second": 40960 / 210,
        "largest_working_set_bytes": 3 * 4096 * 8,
        "largest_median_gb_per_second": 163840 / 1700,
        "largest_to_peak_ratio": (163840 / 1700) / (40960 / 210),
    }
    tied = stream_curves(
        (
            measurement("scale", 256, [100, 100, 100]),
            measurement("scale", 512, [200, 200, 200]),
        )
    )["scale"]
    assert curve_summary(tied)["peak_working_set_bytes"] == 2 * 256 * 8
    assert curve_summary(tied)["largest_to_peak_ratio"] == 1.0
    with pytest.raises(MemoryHierarchyError, match="at least one point"):
        curve_summary([])
    with pytest.raises(TypeError, match="StreamMeasurement"):
        stream_curves(({"kernel": "copy"},))


@pytest.mark.parametrize(
    ("kernels", "elements", "error", "message"),
    [
        ((), (256,), MemoryHierarchyError, "nonempty"),
        (["copy"], (256,), MemoryHierarchyError, "nonempty tuple"),
        (("triad", "copy"), (256,), MemoryHierarchyError, "STREAM order"),
        (("copy", "copy"), (256,), MemoryHierarchyError, "unique"),
        (("fma",), (256,), MemoryHierarchyError, "kernel must be one of"),
        (("copy",), (), MemoryHierarchyError, "1 to 16 points"),
        (("copy",), [256], MemoryHierarchyError, "tuple"),
        (("copy",), (1024, 256), MemoryHierarchyError, "strictly increasing"),
        (("copy",), (256, 256), MemoryHierarchyError, "strictly increasing"),
        (("copy",), (32,), MemoryHierarchyError, "sweep elements"),
        (("copy",), ((1 << 24) + 1,), MemoryHierarchyError, "sweep elements"),
        (("copy",), (256.0,), TypeError, "sweep elements"),
    ],
)
def test_sweep_controls_are_validated(kernels, elements, error, message):
    with pytest.raises(error, match=message):
        validate_sweep(kernels, elements)


def test_host_timer_sweep_reports_positive_finite_rates():
    results = measure_stream_sweep(
        kernels=("copy", "triad"),
        elements=(256, 4096),
        dtype="float32",
        seed=71,
        warmup=1,
        repeats=3,
        target_bytes_per_sample=1 << 16,
    )

    assert {item.timing.timer for item in results} == {"time.perf_counter_ns"}
    for item in results:
        mapping = item.to_mapping()
        assert mapping["median_gb_per_second"] > 0
        assert math.isfinite(mapping["best_gb_per_second"])
        assert mapping["best_gb_per_second"] >= mapping["median_gb_per_second"]
