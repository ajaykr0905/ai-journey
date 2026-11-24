import pytest

from ai_journey.metal_flash_benchmark import (
    FLOPS_PER_PAIR,
    benchmark_point,
    modeled_bytes,
)
from ai_journey.metal_flash_config import FlashPoint, FlashShape, compulsory_bytes
from ai_journey.metal_flash_memory import (
    construction_bytes,
    point_memory,
    side_memory,
)
from ai_journey.metal_flash_plan import SIDES, TORCH_SDPA
from ai_journey.metal_flash_reference import dense_layout
from ai_journey.metal_flash_source import CAUSAL, F1, FULL, N0, MetalFlashError
from test_metal_flash_emulator import FlashEmulator, readings
from test_metal_liger_emulator import ticking_clock

POINT = FlashPoint(1, FlashShape(2, 96), CAUSAL)


def run(backend=None, clock=None):
    return benchmark_point(
        backend or FlashEmulator(),
        seed=81,
        point=POINT,
        inputs="normal",
        check_rows=8,
        warmup=1,
        repeats=3,
        target_bytes_per_sample=1 << 20,
        clock=clock or ticking_clock(),
    )


def test_a_point_times_three_sides_and_records_memory_and_checks():
    mapping = run().to_mapping()
    layout = dense_layout(2, 96)

    assert mapping["side_order"] == list(SIDES)
    assert mapping["mode"] == CAUSAL and mapping["index"] == 1
    assert mapping["pairs"] == 2 * 96 * 97 // 2
    calls = mapping["sides"][N0]["calls_per_sample"]
    assert calls == -(-(1 << 20) // compulsory_bytes(layout))
    for side in SIDES:
        entry = mapping["sides"][side]
        assert entry["flops_per_call"] == FLOPS_PER_PAIR * mapping["pairs"]
        assert len(entry["samples_ns"]) == 3
        assert mapping["checks"][side]["max_bound_ratio"] <= 1
        assert set(mapping["memory"][side]["readings"]) >= {"before", "polls"}
    assert mapping["sides"][TORCH_SDPA]["modeled_bytes_per_call"] is None
    assert mapping["launches"][F1][0]["kernel"] == "flash_f1_causal"


def test_the_traffic_model_counts_the_n_by_n_buffer_and_the_tiles():
    layout = dense_layout(1, 64)

    assert modeled_bytes(N0, layout, FULL) == 16 * 64 * 64 + 16 * 64 * 64
    assert modeled_bytes(F1, layout, FULL) == 8 * 64 * 64 + 16384 * 4
    assert modeled_bytes(F1, layout, CAUSAL) == 8 * 64 * 64 + 16384 * 3
    assert modeled_bytes(TORCH_SDPA, layout, FULL) is None
    with pytest.raises(MetalFlashError):
        modeled_bytes("f2", layout, FULL)


def test_the_construction_bytes_name_the_scratch_of_each_side():
    layout = dense_layout(2, 64)

    assert construction_bytes(N0, layout) == {
        "tensors": 16 * 64 * 128,
        "scratch": 4 * 2 * 64 * 64,
        "index": 4 * (3 * 2 + 2 * 128),
    }
    assert construction_bytes(F1, layout)["scratch"] == 0
    assert construction_bytes(TORCH_SDPA, layout)["scratch"] is None
    record = side_memory(TORCH_SDPA, layout, readings())
    assert record["known_construction_bytes"] == 16 * 64 * 128
    assert record["peak_driver_over_before_bytes"] == 0


@pytest.mark.parametrize(
    "broken",
    [
        {"polls": 0},
        {"before": {"current_bytes": -1, "driver_bytes": 0}},
        {"after_call": {"current_bytes": 1}},
        {"extra": 1},
    ],
)
def test_memory_readings_out_of_form_are_refused(broken):
    entry = {**readings(), **broken}

    with pytest.raises(MetalFlashError):
        side_memory(N0, dense_layout(1, 8), entry)
    with pytest.raises(MetalFlashError, match="three sides"):
        point_memory(dense_layout(1, 8), {N0: readings()})


def test_a_side_whose_output_breaks_its_bound_stops_the_point():
    def far_off(side, output):
        if side == TORCH_SDPA:
            output = output.copy()
            output[0] += 1.0
        return output

    with pytest.raises(MetalFlashError, match="torch_sdpa"):
        run(FlashEmulator(corrupt=far_off))


def test_a_workload_in_another_order_is_refused():
    class Swapped(FlashEmulator):
        def workload(self, inputs, mode):
            workload = super().workload(inputs, mode)
            sides = dict(reversed(list(workload.sides.items())))
            return type(workload)(sides, workload.synchronize, workload.outputs)

    with pytest.raises(MetalFlashError, match="in order"):
        run(Swapped())
