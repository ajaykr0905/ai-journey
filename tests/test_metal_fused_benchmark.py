import time

import numpy
import pytest

from ai_journey.kernel_metal_benchmark import DeviceTiming, RunBudget
from ai_journey.memory_access_benchmark import INJECTED_TIMER, TIMER
from ai_journey.metal_fused_benchmark import (
    VENDOR_UNIT,
    benchmark_point,
    modeled_bytes,
    side_rates,
)
from ai_journey.metal_fused_config import TimingShape, compulsory_bytes
from ai_journey.metal_fused_plan import TORCH_SOFTMAX
from ai_journey.metal_fused_reference import (
    BENCHMARK_STREAM,
    NORMAL,
    dense_layout,
    float32_sha256,
    ragged_layout,
    softmax_inputs,
)
from ai_journey.metal_fused_source import F1, F2, F3, STEP_VARIANTS, U0, MetalFusedError
from test_kernel_metal_runtime import requires_metal
from test_metal_fused_correctness import StandInBackend
from test_metal_fused_runtime import fused_runtime

SIDES = (*STEP_VARIANTS, TORCH_SOFTMAX)


class StepClock:
    """An injected clock that advances by a fixed step at every reading."""

    def __init__(self, step=1000):
        self.now = 0
        self.step = step
        self.readings = 0

    def __call__(self):
        self.now += self.step
        self.readings += 1
        return self.now


def point(backend, shape, clock=None, names=STEP_VARIANTS, index=0, **controls):
    settings = {"warmup": 1, "repeats": 5, "target_bytes_per_sample": 4096}
    settings.update(controls)
    return benchmark_point(
        backend,
        seed=79,
        index=index,
        shape=shape,
        inputs=NORMAL,
        names=names,
        clock=clock or StepClock(),
        **settings,
    )


def test_the_model_counts_three_row_reads_for_u0_and_one_for_a_fused_step():
    layout = dense_layout(4096, 1024)
    elements, rows = layout.elements, layout.rows

    assert modeled_bytes(U0, layout) == 16 * elements + 44 * rows
    for name in (F1, F2, F3):
        assert modeled_bytes(name, layout) == 8 * elements + 8 * rows
    assert compulsory_bytes(layout) == 8 * elements
    assert modeled_bytes(U0, layout) / modeled_bytes(F1, layout) == pytest.approx(
        2.0, rel=0.01
    )
    with pytest.raises(MetalFusedError, match="registered step"):
        modeled_bytes(TORCH_SOFTMAX, layout)


def test_rates_divide_the_bytes_of_a_sample_by_its_median():
    timing = DeviceTiming("x", 4, 1, 5, (100, 200, 300, 400, 500), TIMER)

    rates = side_rates(timing, 1000, 1500)

    assert rates["median_ns_per_call"] == 75.0
    assert rates["effective_gb_per_second"] == 4000 / 300
    assert rates["iqr_effective_gb_per_second"] == [4000 / 400, 4000 / 200]
    assert rates["modeled_gb_per_second"] == 6000 / 300
    assert side_rates(timing, 1000, None)["modeled_gb_per_second"] is None


def test_every_side_is_timed_in_rotation_on_one_batch_size():
    backend = StandInBackend()
    clock = StepClock()
    shape = TimingShape(rows=8, cols=64)

    result = point(backend, shape, clock).to_mapping()

    layout = dense_layout(8, 64)
    calls = -(-4096 // compulsory_bytes(layout))
    assert result["side_order"] == list(SIDES)
    assert result["compulsory_bytes_per_call"] == 8 * 512
    for name in SIDES:
        side = result["sides"][name]
        assert side["calls_per_sample"] == calls == 1
        assert side["samples_ns"] == [1000] * 5
        assert side["modeled_bytes_per_call"] == (
            None if name == TORCH_SOFTMAX else modeled_bytes(name, layout)
        )
        assert backend.calls[name] == (1 + 5) * calls
    assert clock.readings == 2 * 5 * len(SIDES)
    values = softmax_inputs(79, BENCHMARK_STREAM, 0, layout, NORMAL)
    assert result["input_sha256"] == float32_sha256(values)
    assert result["torch_input"] == {"rows": 8, "cols": 64, "padded": False}
    assert result["launch"]["group_size"] == [32]


def test_a_ragged_point_draws_its_lengths_and_pads_the_torch_input():
    result = point(StandInBackend(), TimingShape(rows=6, max_length=300)).to_mapping()

    layout = ragged_layout(79, 0, 6, 300)
    assert result["layout"] == layout.to_mapping()
    assert result["torch_input"] == {
        "rows": 6,
        "cols": layout.longest,
        "padded": True,
    }
    assert result["compulsory_bytes_per_call"] == 8 * layout.elements


def test_the_checks_hold_every_step_to_its_bound_and_torch_to_the_vendor_bound():
    result = point(StandInBackend(), TimingShape(rows=4, cols=100)).to_mapping()

    for name in SIDES:
        check = result["checks"][name]
        assert 0 <= check["max_bound_ratio"] <= 1
        assert check["max_abs_error"] >= 0
    assert result["checks"][TORCH_SOFTMAX]["differing_from_rounded_reference"] == 0
    assert VENDOR_UNIT == 2.0**-16


@pytest.mark.parametrize(
    ("faults", "message"),
    [
        ({"torch_off": True}, "torch_softmax exceeds its bound"),
        ({F3: "off"}, "softmax_f3_online exceeds its bound"),
    ],
)
def test_a_side_whose_latest_output_is_wrong_stops_the_run(faults, message):
    with pytest.raises(MetalFusedError, match=message):
        point(StandInBackend(faults=faults), TimingShape(rows=4, cols=100))


def test_a_point_binds_only_the_admitted_steps_and_names_its_clock():
    result = point(
        StandInBackend(),
        TimingShape(rows=4, cols=64),
        clock=StepClock(),
        names=(U0, F1),
    ).to_mapping()

    assert result["side_order"] == [U0, F1, TORCH_SOFTMAX]
    assert set(result["checks"]) == {U0, F1, TORCH_SOFTMAX}
    first = point(StandInBackend(), TimingShape(rows=4, cols=64))
    assert {timing.timer for timing in first.timings} == {INJECTED_TIMER}


def test_the_budget_stops_a_point_between_samples():
    clock = StepClock(step=10**9)
    budget = RunBudget(5, clock)

    with pytest.raises(ValueError, match="max_runtime_seconds"):
        point(StandInBackend(), TimingShape(rows=4, cols=64), clock, budget=budget)


@requires_metal
def test_a_real_point_is_timed_with_the_host_timer_on_this_gpu():
    shape = TimingShape(rows=64, max_length=500)

    result = benchmark_point(
        fused_runtime(),
        seed=79,
        index=1,
        shape=shape,
        inputs=NORMAL,
        names=STEP_VARIANTS,
        warmup=1,
        repeats=5,
        target_bytes_per_sample=1 << 20,
        clock=time.perf_counter_ns,
    )

    assert {timing.timer for timing in result.timings} == {TIMER}
    mapping = result.to_mapping()
    for name in SIDES:
        assert all(sample > 0 for sample in mapping["sides"][name]["samples_ns"])
        assert mapping["checks"][name]["max_bound_ratio"] <= 1
    assert numpy.isfinite(mapping["sides"][F1]["effective_gb_per_second"])
