import itertools

import pytest

from ai_journey.kernel_metal_benchmark import (
    MAX_CALLS_PER_SAMPLE,
    BenchmarkPoint,
    DeviceTiming,
    RunBudget,
    benchmark_rgb_to_gray,
    benchmark_vector_add,
    calls_per_sample,
    time_pair,
)
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.kernel_metal_runtime import DeviceWorkload
from test_kernel_metal_recorded import RecordedBackend
from test_kernel_metal_runtime import metal_runtime, requires_metal


class LoggedWorkload:
    """Record every launch, synchronize and clock reading in one event list."""

    def __init__(self, agree=True):
        self.events = []
        self.agree = agree
        self.metal = lambda: self.events.append("metal")
        self.torch = lambda: self.events.append("torch")

    def synchronize(self):
        self.events.append("sync")

    def check(self):
        return self.agree

    def clock(self, step=100):
        counter = itertools.count(0, step)

        def read():
            self.events.append("clock")
            return next(counter)

        return read


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


def test_samples_hold_the_fewest_whole_calls_that_reach_the_target():
    assert calls_per_sample(12 * 4096, 1 << 20) == 22
    assert calls_per_sample(1 << 26, 1 << 20) == 1
    assert calls_per_sample(4 * 64 * 64, 1 << 24) == 1024
    with pytest.raises(KernelMetalError, match="at most 4096"):
        calls_per_sample(4, 1 << 20)
    with pytest.raises(KernelMetalError, match="target_bytes_per_sample"):
        calls_per_sample(4, 512)
    assert MAX_CALLS_PER_SAMPLE == 4096


def test_every_sample_is_synchronized_and_the_order_alternates():
    workload = LoggedWorkload()

    metal, torch = time_pair(
        workload, calls_per_sample=2, warmup=1, repeats=3, clock=workload.clock()
    )

    warm = ["sync", "metal", "metal", "sync", "sync", "torch", "torch", "sync"]
    assert workload.events[:8] == warm

    def timed(side):
        return ["sync", "clock", side, side, "sync", "clock"]

    order = ["metal", "torch", "torch", "metal", "metal", "torch"]
    assert workload.events[8:] == [event for side in order for event in timed(side)]
    assert metal == DeviceTiming("metal_kernel", 2, 1, 3, (100, 100, 100), "injected")
    assert torch.implementation == "torch"
    assert metal.to_mapping()["timing"] == {
        "count": 3,
        "min_ns": 100,
        "p25_ns": 100,
        "median_ns": 100,
        "p75_ns": 100,
        "max_ns": 100,
        "iqr_ns": 0,
    }


def test_the_budget_reads_between_samples_and_stops_an_overlong_run():
    workload = LoggedWorkload()
    clock = workload.clock(step=300_000_000)
    budget = RunBudget(1, clock)

    with pytest.raises(KernelMetalError, match="max_runtime_seconds=1"):
        time_pair(
            workload,
            calls_per_sample=1,
            warmup=0,
            repeats=3,
            clock=clock,
            budget=budget,
        )
    assert workload.events.count("metal") == 1
    assert workload.events.count("torch") == 1


@pytest.mark.parametrize(
    ("readings", "message"),
    [
        ([0, 0], "did not advance"),
        ([5, 3], "monotonic"),
        ([0, 1.5], "integer nanoseconds"),
        ([-1, 2], "nonnegative"),
    ],
)
def test_clock_contract_violations_are_rejected(readings, message):
    clock = iter(readings + [10**9] * 20).__next__

    with pytest.raises(KernelMetalError, match=message):
        time_pair(
            LoggedWorkload(), calls_per_sample=1, warmup=0, repeats=3, clock=clock
        )


def test_vector_add_points_record_samples_rates_and_the_time_ratio():
    backend = RecordedBackend()

    points = benchmark_vector_add(
        backend,
        seed=73,
        lengths=(4096, 65537),
        group_size=256,
        warmup=1,
        repeats=3,
        target_bytes_per_sample=1 << 20,
        clock=scripted_clock(),
    )

    assert [point.launch.geometry.extent for point in points] == [(4096,), (65537,)]
    mapping = points[1].to_mapping()
    assert (mapping["kernel"], mapping["length"]) == ("vector_add", 65537)
    assert mapping["bytes_per_call"] == 12 * 65537
    assert mapping["launch"]["padding_threads"] == 255
    metal = mapping["metal_kernel"]
    assert (metal["calls_per_sample"], metal["samples_ns"]) == (2, [1000, 1000, 1000])
    assert metal["bytes_per_sample"] == 2 * 12 * 65537
    assert metal["median_ns_per_call"] == 500.0
    assert metal["iqr_ns_per_call"] == 0.0
    assert metal["median_gb_per_second"] == 2 * 12 * 65537 / 1000
    assert metal["best_gb_per_second"] == metal["median_gb_per_second"]
    assert mapping["torch"]["implementation"] == "torch"
    assert mapping["torch_to_metal_time_ratio"] == 1.0
    assert points[0].to_mapping()["metal_kernel"]["calls_per_sample"] == 22
    assert len(backend.launches) == 2


def test_grayscale_points_name_rows_and_columns_and_count_four_bytes():
    points = benchmark_rgb_to_gray(
        RecordedBackend(),
        seed=73,
        shapes=((17, 33), (64, 48)),
        group_size=(16, 16),
        warmup=0,
        repeats=3,
        target_bytes_per_sample=1 << 14,
        clock=scripted_clock(),
    )

    first = points[0].to_mapping()
    assert (first["height"], first["width"]) == (17, 33)
    assert first["bytes_per_call"] == 4 * 17 * 33
    assert first["launch"]["threads"] == [48, 32]
    assert first["metal_kernel"]["calls_per_sample"] == 8
    assert points[1].bytes_per_call == 4 * 64 * 48
    assert isinstance(points[1], BenchmarkPoint)


def test_points_whose_timed_outputs_disagree_stop_the_run():
    class Disagrees(RecordedBackend):
        def vector_add_workload(self, first, second, launch):
            workload = super().vector_add_workload(first, second, launch)
            return DeviceWorkload(
                workload.metal, workload.torch, workload.synchronize, lambda: False
            )

    with pytest.raises(KernelMetalError, match=r"\[4096\]: the timed Metal and"):
        benchmark_vector_add(
            Disagrees(),
            seed=73,
            lengths=(4096,),
            group_size=256,
            warmup=0,
            repeats=3,
            target_bytes_per_sample=1 << 16,
            clock=scripted_clock(),
        )
    with pytest.raises(KernelMetalError, match="1 to 16 points"):
        benchmark_vector_add(
            RecordedBackend(),
            seed=73,
            lengths=[4096],
            group_size=256,
            warmup=0,
            repeats=3,
            target_bytes_per_sample=1 << 16,
        )


@requires_metal
def test_both_sides_time_on_this_gpu_with_the_host_timer():
    vector = benchmark_vector_add(
        metal_runtime(),
        seed=73,
        lengths=(4097, 1 << 18),
        group_size=256,
        warmup=1,
        repeats=3,
        target_bytes_per_sample=1 << 20,
    )
    gray = benchmark_rgb_to_gray(
        metal_runtime(),
        seed=73,
        shapes=((63, 65),),
        group_size=(16, 16),
        warmup=1,
        repeats=3,
        target_bytes_per_sample=1 << 16,
    )

    for point in (*vector, *gray):
        for timing in (point.metal, point.torch):
            assert timing.timer == "time.perf_counter_ns"
            assert len(timing.samples_ns) == 3
            assert all(sample > 0 for sample in timing.samples_ns)
        assert point.torch_to_metal_time_ratio > 0
