import time

import pytest

from ai_journey.gpu_timing_benchmark import SCHEDULE, SideTiming, time_sides
from ai_journey.gpu_timing_clocks import (
    CLOCKS,
    DISPATCH,
    EVENT,
    SYNCHRONIZED,
    ClockSamples,
    GpuTimingError,
)
from ai_journey.kernel_metal_benchmark import RunBudget
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.memory_access_benchmark import INJECTED_TIMER, TIMER


class FakeDevice:
    """A virtual host clock and GPU queue: launches cost host time, then GPU time."""

    def __init__(self, dispatch_ns=1_000, gpu_ns=None, wake_ns=500):
        self.now = 0
        self.busy_until = 0
        self.dispatch_ns = dispatch_ns
        self.gpu_ns = gpu_ns or {}
        self.wake_ns = wake_ns
        self.log = []

    def clock(self):
        return self.now

    def side(self, name):
        def call():
            self.log.append(("launch", name))
            self.now += self.dispatch_ns
            start = max(self.now, self.busy_until)
            self.busy_until = start + self.gpu_ns.get(name, 2_000)

        return call

    def synchronize(self):
        self.log.append(("sync",))
        self.now = max(self.now, self.busy_until) + self.wake_ns

    def event_pair(self):
        device = self

        class Pair:
            def start(self):
                device.log.append(("event", "start"))
                self.began = max(device.now, device.busy_until)

            def stop(self):
                device.log.append(("event", "stop"))
                self.ended = device.busy_until

            def elapsed_ms(self):
                return (self.ended - self.began) / 1e6

        return Pair()


def run(device, names=("a", "b"), warmup=1, repeats=3, **kwargs):
    return time_sides(
        {name: device.side(name) for name in names},
        synchronize=device.synchronize,
        event_pair=device.event_pair,
        warmup=warmup,
        repeats=repeats,
        clock=device.clock,
        **kwargs,
    )


def test_each_clock_measures_its_own_interval():
    device = FakeDevice(dispatch_ns=1_000, gpu_ns={"a": 7_000}, wake_ns=500)

    (timing,) = run(device, names=("a",))

    # The fake GPU starts the work once the launch is encoded.
    assert timing.samples(EVENT).samples_ns == (8_000,) * 3
    assert timing.samples(EVENT).elapsed_ms == (0.008,) * 3
    # Dispatch, then the GPU work, then the wake after it.
    assert timing.samples(SYNCHRONIZED).samples_ns == (8_500,) * 3
    assert timing.samples(DISPATCH).samples_ns == (1_000,) * 3
    assert timing.timer == INJECTED_TIMER


def test_every_sample_drains_the_queue_and_holds_one_launch():
    device = FakeDevice()

    run(device, names=("a",), warmup=0, repeats=3)

    entries = device.log
    launches = [index for index, entry in enumerate(entries) if entry[0] == "launch"]
    assert len(launches) == 3 * len(CLOCKS)
    for index in launches:
        before = [entry for entry in entries[:index] if entry[0] != "event"]
        assert before[-1] == ("sync",)
    assert entries[-1] == ("sync",)


def test_warmups_run_first_and_the_order_rotates_over_sides_and_clocks():
    device = FakeDevice()

    run(device, names=("a", "b", "c"), warmup=2, repeats=3)

    launches = [entry[1] for entry in device.log if entry[0] == "launch"]
    warm = 2 * len(CLOCKS)
    assert launches[: 3 * warm] == ["a"] * warm + ["b"] * warm + ["c"] * warm
    timed = launches[3 * warm :]
    per_repeat = 3 * len(CLOCKS)
    firsts = [timed[r * per_repeat] for r in range(3)]
    assert firsts == ["a", "b", "c"]
    assert "shifted by r" in SCHEDULE


def test_the_clock_order_inside_a_side_rotates_too():
    device = FakeDevice()
    seen = []
    original = device.event_pair

    def tagged():
        pair = original()
        start = pair.start

        def tagged_start():
            seen.append(len([e for e in device.log if e[0] == "launch"]))
            start()

        pair.start = tagged_start
        return pair

    time_sides(
        {"a": device.side("a")},
        synchronize=device.synchronize,
        event_pair=tagged,
        warmup=0,
        repeats=3,
        clock=device.clock,
    )

    # The event sample is first, then last, then second in its repeat.
    assert seen == [0, 3 + 2, 6 + 1]


def test_the_host_timer_is_named_only_for_perf_counter_ns():
    device = FakeDevice()
    sides = {"a": device.side("a")}

    (timing,) = time_sides(
        sides,
        synchronize=device.synchronize,
        event_pair=device.event_pair,
        warmup=0,
        repeats=3,
        clock=time.perf_counter_ns,
    )

    assert timing.timer == TIMER


def test_a_stalled_clock_is_refused():
    device = FakeDevice(dispatch_ns=0)

    with pytest.raises(GpuTimingError, match="did not advance"):
        run(device, names=("a",), warmup=0)


def test_a_zero_event_interval_is_refused():
    device = FakeDevice()

    class Stuck:
        def start(self):
            pass

        def stop(self):
            pass

        def elapsed_ms(self):
            return 0.0

    with pytest.raises(GpuTimingError, match="MPS event interval"):
        time_sides(
            {"a": device.side("a")},
            synchronize=device.synchronize,
            event_pair=Stuck,
            warmup=0,
            repeats=3,
            clock=device.clock,
        )


def test_the_budget_stops_a_long_run_between_samples():
    device = FakeDevice(gpu_ns={"a": 400_000_000})
    budget = RunBudget(1, device.clock)

    with pytest.raises(KernelMetalError, match="max_runtime_seconds=1"):
        run(device, names=("a",), warmup=0, repeats=5, budget=budget)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"warmup": 11}, ValueError),
        ({"repeats": 2}, ValueError),
        ({"repeats": 3.0}, TypeError),
        ({"budget": object()}, TypeError),
        ({"names": ()}, GpuTimingError),
    ],
)
def test_bad_controls_are_refused(kwargs, error):
    with pytest.raises(error):
        run(FakeDevice(), **kwargs)


def timing(dispatch=(1, 1, 1), synchronized=(5, 5, 5), repeats=3):
    return SideTiming(
        "a",
        0,
        repeats,
        (
            ClockSamples.from_event_ms((0.001,) * 3),
            ClockSamples(SYNCHRONIZED, synchronized),
            ClockSamples(DISPATCH, dispatch),
        ),
        TIMER,
    )


def test_a_dispatch_median_above_the_synchronized_median_is_refused():
    assert timing().median_ns(DISPATCH) == 1
    with pytest.raises(GpuTimingError, match="dispatch median exceeds"):
        timing(dispatch=(9, 9, 9))


def test_a_side_needs_every_clock_in_order_with_one_sample_per_repeat():
    with pytest.raises(GpuTimingError, match="one sample per repeat"):
        timing(repeats=4)
    with pytest.raises(GpuTimingError, match="needs samples of"):
        SideTiming("a", 0, 3, (ClockSamples(SYNCHRONIZED, (1, 1, 1)),), TIMER)


def test_the_mapping_keeps_raw_samples_by_clock():
    mapping = timing().to_mapping()

    assert list(mapping["clocks"]) == list(CLOCKS)
    assert mapping["clocks"][EVENT]["elapsed_ms"] == [0.001] * 3
    assert mapping["clocks"][DISPATCH]["samples_ns"] == [1, 1, 1]
