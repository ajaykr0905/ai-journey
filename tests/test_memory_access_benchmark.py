import time

import pytest

from ai_journey.memory_access_benchmark import (
    MAX_REPEATS,
    PERCENTILE_METHOD,
    TimingSummary,
    gflops_per_second,
    summarize_timings,
    time_repeats,
    timer_name,
)
from ai_journey.memory_access_pattern import MemoryAccessError


class FakeClock:
    """Return scripted nanosecond readings and record when each one happens."""

    def __init__(self, readings, events):
        self.readings = iter(readings)
        self.events = events

    def __call__(self):
        value = next(self.readings)
        self.events.append(("clock", value))
        return value


def test_nearest_rank_summary_reports_median_min_and_iqr_without_interpolation():
    odd = summarize_timings((50, 10, 40, 20, 30))
    even = summarize_timings((40, 10, 30, 20))

    assert odd == TimingSummary(
        count=5, min_ns=10, p25_ns=20, median_ns=30, p75_ns=40, max_ns=50
    )
    assert odd.iqr_ns == 20
    assert (even.p25_ns, even.median_ns, even.p75_ns) == (10, 20, 30)
    assert summarize_timings((7,)).to_mapping() == {
        "count": 1,
        "min_ns": 7,
        "p25_ns": 7,
        "median_ns": 7,
        "p75_ns": 7,
        "max_ns": 7,
        "iqr_ns": 0,
    }
    assert "nearest rank" in PERCENTILE_METHOD
    assert "no interpolation" in PERCENTILE_METHOD


@pytest.mark.parametrize(
    "samples", [(), [1, 2], (0,), (-5,), (True,), (1.5,), (3, "4")]
)
def test_summary_rejects_samples_that_are_not_positive_integer_nanoseconds(samples):
    with pytest.raises(MemoryAccessError, match="samples_ns"):
        summarize_timings(samples)


def test_injected_clock_times_only_measured_calls_and_inspects_after_stopping():
    events = []
    clock = FakeClock([100, 250, 250, 400, 1_000, 1_300], events)

    def operation():
        events.append(("call", None))
        return len(events)

    samples = time_repeats(
        operation,
        warmup=2,
        repeats=3,
        clock=clock,
        inspect=lambda result: events.append(("inspect", result)),
    )

    assert samples == (150, 150, 300)
    kinds = [kind for kind, _ in events]
    assert kinds[:4] == ["call", "inspect", "call", "inspect"]
    assert kinds[4:] == ["clock", "call", "clock", "inspect"] * 3


@pytest.mark.parametrize(
    ("readings", "message"),
    [
        ([10, 5], "monotonic"),
        ([5, 5], "did not advance"),
        ([5, 7.0], "integer nanoseconds"),
        ([True, 9], "integer nanoseconds"),
        ([-1, 9], "integer nanoseconds"),
        ([1, 3, 2, 4], "monotonic"),
    ],
)
def test_clock_must_be_monotonic_integer_and_advance_across_each_call(
    readings, message
):
    with pytest.raises(MemoryAccessError, match=message):
        time_repeats(
            lambda: None,
            warmup=0,
            repeats=len(readings) // 2,
            clock=FakeClock(readings, []),
        )


def test_only_the_host_timer_is_named_as_a_measurement_clock(monkeypatch):
    host = time.perf_counter_ns

    assert timer_name(host) == "time.perf_counter_ns"
    assert timer_name(lambda: host()) == "injected"
    assert timer_name(FakeClock([1, 2], [])) == "injected"
    monkeypatch.setattr(time, "perf_counter_ns", lambda: 5)
    assert timer_name(time.perf_counter_ns) == "injected"


def test_inspection_failures_propagate_instead_of_being_timed():
    def reject(result):
        raise RuntimeError(f"bad result {result}")

    with pytest.raises(RuntimeError, match="bad result 3"):
        time_repeats(
            lambda: 3, warmup=0, repeats=1, clock=FakeClock([1, 2], []), inspect=reject
        )


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"warmup": 0, "repeats": 0}, MemoryAccessError),
        ({"warmup": 0, "repeats": MAX_REPEATS + 1}, MemoryAccessError),
        ({"warmup": 11, "repeats": 1}, MemoryAccessError),
        ({"warmup": -1, "repeats": 1}, MemoryAccessError),
        ({"warmup": 0, "repeats": True}, TypeError),
        ({"warmup": 0, "repeats": 1, "clock": 5}, TypeError),
        ({"warmup": 0, "repeats": 1, "inspect": "print"}, TypeError),
    ],
)
def test_timing_budget_is_bounded(arguments, error):
    with pytest.raises(error):
        time_repeats(lambda: None, **arguments)


def test_gflops_divides_flops_by_nanoseconds():
    assert gflops_per_second(2_000, 1_000) == 2.0
    assert gflops_per_second(2 * 512**3, 1_250_000) == 214.7483648
    with pytest.raises(MemoryAccessError, match="elapsed_ns"):
        gflops_per_second(10, 0)
    with pytest.raises(TypeError, match="flops"):
        gflops_per_second(1.0, 10)
