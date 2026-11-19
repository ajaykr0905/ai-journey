import functools

import numpy
import pytest

from ai_journey.gpu_tiled_blocksim import (
    PHASE_STEPS,
    SharedHazards,
    require_steps,
    run_tiled_blocks,
)
from ai_journey.gpu_tiled_cuda import TILES, GpuTiledError
from ai_journey.matmul_metal_mapping import MatmulMetalError
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
)
from ai_journey.matmul_model_kernels import X_COLUMNS, matmul_launch
from ai_journey.memory_access_tiling import MatmulShape, TiledMatmulTraffic

# The seed and correctness shapes of config/day-74-matmul-metal.json, in order.
DAY_74_SEED = 74
DAY_74_SHAPES = tuple(
    MatmulShape(*triple)
    for triple in (
        (1, 1, 1),
        (17, 33, 9),
        (32, 32, 16),
        (31, 47, 129),
        (129, 65, 1000),
        (255, 257, 511),
        (64, 48, 4096),
        (3, 1000, 2048),
        (1000, 999, 100),
    )
)
RAGGED = MatmulShape(37, 45, 29)


@functools.cache
def day_74_case(index):
    left, right = matmul_inputs(
        DAY_74_SEED, CORRECTNESS_STREAM, index, DAY_74_SHAPES[index]
    )
    return left, right, fma_replay(left, right)


def ragged_operands(seed=0):
    generator = numpy.random.default_rng(seed)
    left = generator.standard_normal((RAGGED.m, RAGGED.k), dtype=numpy.float32)
    right = generator.standard_normal((RAGGED.k, RAGGED.n), dtype=numpy.float32)
    return left, right


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("index", range(len(DAY_74_SHAPES)))
def test_each_day_74_case_matches_the_fma_replay_bit_for_bit(index, tile):
    left, right, replay = day_74_case(index)
    shape = DAY_74_SHAPES[index]

    run = run_tiled_blocks(left, right, tile=tile)

    assert run.output.dtype == numpy.float32
    assert numpy.array_equal(run.output.view(numpy.uint32), replay.view(numpy.uint32))
    assert run.output_sha256 == matrix_sha256(replay)
    assert run.matches_traffic
    assert run.hazards.race_free and run.stored_once
    assert run.launch == matmul_launch(shape, (tile, tile), X_COLUMNS)
    assert run.inner_steps == shape.k
    assert run.phases == -(-shape.k // tile)
    assert run.hazards.barriers == 2 * run.phases


def test_the_counted_loads_are_the_traffic_model_of_a_ragged_launch():
    left, right = ragged_operands()
    run = run_tiled_blocks(left, right, tile=16)
    model = TiledMatmulTraffic(RAGGED, tile=16)

    # Each of the 3 block columns reads all of A, and each of the 3 block
    # rows reads all of B; the rest of the 2 * 16 * 16 * 2 * 9 slots are 0.
    assert (run.a_loads, run.b_loads) == (3 * 37 * 29, 3 * 29 * 45)
    assert run.tile_slot_writes == 9216
    assert run.zero_filled_slots == 9216 - 3219 - 3915
    assert (run.a_loads, run.b_loads) == (model.useful_a_loads, model.useful_b_loads)
    assert run.zero_filled_slots == model.padded_slots
    assert run.stores == model.store_elements == 37 * 45
    assert run.to_mapping()["global_loads"] == {
        "a": 3219,
        "b": 3915,
        "zero_filled_slots": 2082,
        "tile_slot_writes": 9216,
    }


def test_a_dropped_barrier_is_a_counted_hazard_with_the_same_values():
    left, right = ragged_operands()
    pinned = run_tiled_blocks(left, right, tile=16)
    tile, k, phases = 16, RAGGED.k, 2

    no_first = run_tiled_blocks(left, right, tile=16, steps=("load", "fma", "barrier"))
    no_second = run_tiled_blocks(left, right, tile=16, steps=("load", "barrier", "fma"))
    none = run_tiled_blocks(left, right, tile=16, steps=("load", "fma"))
    leading = run_tiled_blocks(
        left, right, tile=16, steps=("barrier", "load", "barrier", "fma")
    )

    # Each slot that a phase reads has T - 1 readers besides its writer, in
    # both tiles. Without the first barrier every read races its load; without
    # the second every load after the first races the reads before it.
    assert pinned.hazards == SharedHazards(4, 0, 0)
    assert no_first.hazards == SharedHazards(2, 2 * tile * (tile - 1) * k, 0)
    assert no_second.hazards == SharedHazards(
        2, 0, 2 * tile * (tile - 1) * tile * (phases - 1)
    )
    assert none.hazards == SharedHazards(0, 13920, 7680)
    assert leading.hazards == SharedHazards(4, 0, 0)
    for run in (no_first, no_second, none, leading):
        assert run.output_sha256 == pinned.output_sha256
    assert not no_first.hazards.race_free and leading.hazards.race_free
    assert no_first.to_mapping()["hazards"] == {
        "barriers": 2,
        "read_after_write": 13920,
        "write_after_read": 0,
        "race_free": False,
    }


def test_one_phase_needs_no_trailing_barrier():
    left, right, _ = day_74_case(2)

    run = run_tiled_blocks(left, right, tile=16, steps=("load", "barrier", "fma"))

    assert run.phases == 1 and run.hazards == SharedHazards(1, 0, 0)


def test_the_mapping_names_the_launch_steps_and_digest():
    left, right = ragged_operands()
    mapping = run_tiled_blocks(left, right, tile=8).to_mapping()

    assert mapping["tile"] == 8
    assert mapping["shape"] == {"m": 37, "n": 45, "k": 29}
    assert mapping["launch"]["grid"] == [6, 5, 1]
    assert mapping["launch"]["block"] == [8, 8, 1]
    assert mapping["steps"] == list(PHASE_STEPS)
    assert (mapping["phases"], mapping["inner_steps"]) == (4, 29)
    assert mapping["stores"] == 37 * 45 and mapping["stored_once"] is True
    assert mapping["matches_traffic"] is True
    assert mapping["output_sha256"] == matrix_sha256(fma_replay(left, right))


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        (["load", "barrier", "fma"], "tuple"),
        (("load",), "tuple"),
        (("load", "barrier") * 4 + ("fma",), "tuple"),
        (("load", "sync", "fma"), "one of"),
        (("load", "fma", "fma"), "exactly one"),
        (("barrier", "fma"), "exactly one"),
        (("fma", "barrier", "load"), "follow the load"),
    ],
)
def test_schedules_need_one_load_before_one_fma(steps, message):
    with pytest.raises(GpuTiledError, match=message):
        require_steps(steps)
    assert require_steps(PHASE_STEPS) == PHASE_STEPS


def test_operands_and_tiles_are_validated():
    left, right = ragged_operands()

    with pytest.raises(GpuTiledError, match="tile must be one of"):
        run_tiled_blocks(left, right, tile=4)
    with pytest.raises(TypeError, match="tile must be an integer"):
        run_tiled_blocks(left, right, tile=True)
    with pytest.raises(MatmulMetalError, match="conformable 2D float32"):
        run_tiled_blocks(left.astype(numpy.float64), right, tile=8)
    with pytest.raises(MatmulMetalError, match="conformable 2D float32"):
        run_tiled_blocks(left, right.T.copy(), tile=8)
