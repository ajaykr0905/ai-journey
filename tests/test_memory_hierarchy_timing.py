import time

import pytest

from ai_journey.memory_hierarchy_stream import MemoryHierarchyError
from ai_journey.memory_hierarchy_timing import (
    MAX_CALLS_PER_SAMPLE,
    BatchTiming,
    RunBudget,
    calls_per_sample,
    per_nanosecond,
    per_second,
    time_batches,
)


class ScriptedClock:
    """Return scripted nanosecond readings and log each one between events."""

    def __init__(self, readings, events):
        self.readings = iter(readings)
        self.events = events

    def __call__(self):
        value = next(self.readings)
        self.events.append(("clock", value))
        return value


def test_samples_are_sized_by_bytes_to_the_fewest_whole_calls():
    assert calls_per_sample(1000, 4096) == 5
    assert calls_per_sample(4096, 4096) == 1
    assert calls_per_sample(1 << 30, 4096) == 1
    assert calls_per_sample(16, 1 << 20) == MAX_CALLS_PER_SAMPLE
    with pytest.raises(MemoryHierarchyError, match="at most 65536"):
        calls_per_sample(8, 1 << 20)
    with pytest.raises(MemoryHierarchyError, match="target_bytes_per_sample"):
        calls_per_sample(8, 512)
    with pytest.raises(TypeError, match="bytes_per_call"):
        calls_per_sample(8.0, 4096)


def test_each_sample_times_a_whole_batch_and_the_budget_reads_between_them():
    events = []
    clock = ScriptedClock(range(0, 10_000, 100), events)
    budget = RunBudget(1, clock)

    def call():
        events.append(("call", None))

    timing = time_batches(
        call, calls_per_sample=2, warmup=1, repeats=3, clock=clock, budget=budget
    )

    assert timing == BatchTiming(
        calls_per_sample=2,
        warmup=1,
        repeats=3,
        samples_ns=(100, 100, 100),
        timer="injected",
    )
    kinds = [kind for kind, _ in events]
    assert kinds[:4] == ["clock", "call", "call", "clock"]
    assert kinds[4:] == ["clock", "call", "call", "clock", "clock"] * 3
    assert budget.elapsed_ns == 1000
    assert timing.to_mapping() == {
        "calls_per_sample": 2,
        "warmup": 1,
        "repeats": 3,
        "samples_ns": [100, 100, 100],
        "timing": {
            "count": 3,
            "min_ns": 100,
            "p25_ns": 100,
            "median_ns": 100,
            "p75_ns": 100,
            "max_ns": 100,
            "iqr_ns": 0,
        },
    }


def test_the_budget_stops_a_run_after_the_sample_that_exceeds_it():
    second = 1_000_000_000
    readings = [0, 10, 20, 400_000_000, 600_000_000, 900_000_000, second + 1]
    clock = ScriptedClock(readings, [])
    budget = RunBudget(1, clock)
    calls = []

    with pytest.raises(MemoryHierarchyError, match="max_runtime_seconds=1"):
        time_batches(
            lambda: calls.append(1),
            calls_per_sample=1,
            warmup=0,
            repeats=5,
            clock=clock,
            budget=budget,
        )

    assert len(calls) == 2
    assert budget.elapsed_ns == second + 1


@pytest.mark.parametrize(
    ("readings", "message"),
    [
        ([10, 5], "monotonic"),
        ([5, 5], "did not advance"),
        ([5, 7.0], "integer nanoseconds"),
    ],
)
def test_timing_core_clock_errors_surface_as_lab_errors(readings, message):
    with pytest.raises(MemoryHierarchyError, match=message):
        time_batches(
            lambda: None,
            calls_per_sample=1,
            warmup=0,
            repeats=3,
            clock=ScriptedClock(readings, []),
        )


def test_budget_validates_its_own_clock_readings_and_limit():
    with pytest.raises(MemoryHierarchyError, match="integer nanoseconds"):
        RunBudget(1, lambda: -1)
    budget = RunBudget(1, ScriptedClock([50, 40], []))
    with pytest.raises(MemoryHierarchyError, match="monotonic"):
        budget.check()
    with pytest.raises(MemoryHierarchyError, match="max_runtime_seconds"):
        RunBudget(0, time.perf_counter_ns)
    with pytest.raises(TypeError, match="max_runtime_seconds"):
        RunBudget(True, time.perf_counter_ns)
    with pytest.raises(TypeError, match="clock"):
        RunBudget(1, 5)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"repeats": 2}, MemoryHierarchyError, "repeats"),
        ({"repeats": 51}, MemoryHierarchyError, "repeats"),
        ({"warmup": 11}, MemoryHierarchyError, "warmup"),
        ({"calls_per_sample": 0}, MemoryHierarchyError, "calls_per_sample"),
        ({"calls_per_sample": 1.0}, TypeError, "calls_per_sample"),
        ({"budget": "soon"}, TypeError, "budget"),
    ],
)
def test_batch_budgets_are_bounded(changes, error, message):
    arguments = {"calls_per_sample": 1, "warmup": 0, "repeats": 3, **changes}
    with pytest.raises(error, match=message):
        time_batches(lambda: None, **arguments)


def test_host_timer_samples_are_positive_and_labeled_as_measurements():
    timing = time_batches(
        lambda: sum(range(64)), calls_per_sample=8, warmup=1, repeats=3
    )

    assert timing.timer == "time.perf_counter_ns"
    assert all(type(sample) is int and sample > 0 for sample in timing.samples_ns)
    assert timing.summary.min_ns <= timing.summary.median_ns


def test_rates_divide_exact_integers_once():
    assert per_nanosecond(2_000, 1_000) == 2.0
    assert per_second(3, 2) == 1.5e9
    big = 10**15 + 1
    assert per_second(big, 3) == big * 1_000_000_000 / 3
    assert per_nanosecond(0, 7) == 0.0
    with pytest.raises(MemoryHierarchyError, match="elapsed_ns"):
        per_second(1, 0)
    with pytest.raises(TypeError, match="count"):
        per_nanosecond(1.5, 3)
