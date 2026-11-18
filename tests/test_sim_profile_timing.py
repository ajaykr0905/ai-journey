import itertools
from pathlib import Path

import pytest

from ai_journey.kernel_launch_config import KernelLaunchError
from ai_journey.memory_access_benchmark import INJECTED_TIMER, TIMER
from ai_journey.sim_profile_config import SimProfileError, load_config
from ai_journey.sim_profile_timing import (
    WorkloadTiming,
    host_environment,
    read_load_average,
    time_workload,
)
from ai_journey.sim_profile_workloads import Workload, build_workloads

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/day-75-sim-profile.json"


@pytest.fixture(scope="module")
def vector():
    return build_workloads(load_config(CONFIG))[0]


def stepping_clock(step):
    counter = itertools.count(0, step)
    return lambda: next(counter)


def test_an_injected_clock_gives_exact_synthetic_samples(vector):
    timing = time_workload(vector, warmup=1, repeats=5, clock=stepping_clock(7))

    assert timing.timer == INJECTED_TIMER
    assert timing.samples_ns == (7,) * 5
    assert timing.to_mapping() == {
        "warmup": 1,
        "repeats": 5,
        "samples_ns": [7] * 5,
        "timing": {
            "count": 5,
            "min_ns": 7,
            "p25_ns": 7,
            "median_ns": 7,
            "p75_ns": 7,
            "max_ns": 7,
            "iqr_ns": 0,
        },
    }


def test_the_host_timer_is_named_a_measurement(vector):
    timing = time_workload(vector, warmup=0, repeats=5)

    assert timing.timer == TIMER
    assert all(sample > 0 for sample in timing.samples_ns)
    summary = timing.summary
    assert summary.p25_ns <= summary.median_ns <= summary.p75_ns
    assert summary.iqr_ns == summary.p75_ns - summary.p25_ns


def test_every_pass_is_checked_against_its_reference(vector):
    calls = []

    def wrong():
        calls.append(1)
        return build_workloads(load_config(CONFIG))[1].run()

    bad = Workload(
        name=vector.name,
        config=vector.config,
        output_dtype=vector.output_dtype,
        reference_sha256=vector.reference_sha256,
        run=wrong,
    )
    with pytest.raises(SimProfileError, match="differs from its reference"):
        time_workload(bad, warmup=1, repeats=5, clock=stepping_clock(1))
    assert calls == [1]


def test_timing_bounds_and_clocks_are_enforced(vector):
    with pytest.raises(KernelLaunchError, match="repeats"):
        time_workload(vector, warmup=0, repeats=4)
    with pytest.raises(KernelLaunchError, match="warmup"):
        time_workload(vector, warmup=11, repeats=5)
    with pytest.raises(TypeError, match="Workload"):
        time_workload("vector", warmup=0, repeats=5)
    with pytest.raises(SimProfileError, match="did not advance"):
        time_workload(vector, warmup=0, repeats=5, clock=lambda: 5)
    backwards = iter([10, 5])
    with pytest.raises(SimProfileError, match="monotonic"):
        time_workload(vector, warmup=0, repeats=5, clock=lambda: next(backwards))
    with pytest.raises(Exception, match="positive integer"):
        WorkloadTiming("vector_add", 0, 1, (0,), TIMER).summary


def test_load_averages_and_environment_are_strict_tokens():
    assert read_load_average(lambda: (1.5, 2.0, 0.0)) == [1.5, 2.0, 0.0]
    assert len(read_load_average()) == 3
    for bad in ((1.0, 2.0), (1.0, 2.0, float("nan")), (1, 2.0, 3.0), [1.0] * 3):
        with pytest.raises(SimProfileError, match="three nonnegative floats"):
            read_load_average(lambda bad=bad: bad)
    environment = host_environment()
    assert set(environment) == {
        "python_version",
        "python_implementation",
        "platform_system",
        "platform_machine",
    }
    assert environment["python_implementation"] == "CPython"
