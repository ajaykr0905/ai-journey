import pytest

from ai_journey.memory_access_pattern import MemoryAccessError
from ai_journey.memory_access_tiling import (
    MatmulShape,
    NaiveMatmulTraffic,
    TiledMatmulTraffic,
)


def enumerate_tiled_slots(shape: MatmulShape, tile: int) -> tuple[int, int, int]:
    """Walk every block, phase and tile slot; return (slots, A loads, B loads)."""

    slots = useful_a = useful_b = 0
    for block_row in range(-(-shape.m // tile)):
        for block_column in range(-(-shape.n // tile)):
            for phase in range(-(-shape.k // tile)):
                for row in range(tile):
                    for column in range(tile):
                        slots += 2
                        useful_a += (
                            block_row * tile + row < shape.m
                            and phase * tile + column < shape.k
                        )
                        useful_b += (
                            phase * tile + row < shape.k
                            and block_column * tile + column < shape.n
                        )
    return slots, useful_a, useful_b


def test_naive_matmul_loads_both_operands_for_every_product_term():
    traffic = NaiveMatmulTraffic(MatmulShape(4, 4, 4))

    assert traffic.load_elements == 128
    assert traffic.store_elements == 16
    assert traffic.flops == 128
    assert (traffic.load_bytes, traffic.store_bytes) == (512, 64)
    assert traffic.total_bytes == 576
    assert traffic.arithmetic_intensity == 128 / 576


def test_divisible_tiling_divides_global_loads_by_the_tile_width():
    shape = MatmulShape(64, 64, 64)
    tiled = TiledMatmulTraffic(shape, tile=16)

    assert (tiled.block_rows, tiled.block_columns, tiled.blocks) == (4, 4, 16)
    assert tiled.phases == 4
    assert tiled.tile_slot_loads == 2 * 16 * 16 * 4 * 16
    assert tiled.useful_loads == tiled.tile_slot_loads == 2 * 64**3 // 16
    assert tiled.padded_slots == 0
    assert tiled.reuse_factor == 16
    assert tiled.shared_memory_bytes_per_block == 2048
    assert tiled.threads_per_block == 256
    assert tiled.store_bytes == NaiveMatmulTraffic(shape).store_bytes
    assert tiled.arithmetic_intensity == 524_288 / (32_768 * 4 + 4_096 * 4)


def test_partial_edge_tiles_split_slots_into_useful_and_padded_loads():
    shape = MatmulShape(10, 6, 7)
    tiled = TiledMatmulTraffic(shape, tile=4)

    assert (tiled.block_rows, tiled.block_columns, tiled.phases) == (3, 2, 2)
    assert tiled.tile_slot_loads == 384
    assert (tiled.useful_a_loads, tiled.useful_b_loads) == (140, 126)
    assert tiled.padded_slots == 118
    assert enumerate_tiled_slots(shape, 4) == (384, 140, 126)


@pytest.mark.parametrize(
    ("m", "n", "k", "tile"),
    [(1, 1, 1, 1), (5, 3, 2, 32), (33, 17, 9, 8), (31, 65, 40, 16), (7, 7, 7, 3)],
)
def test_closed_form_tile_accounting_matches_slot_enumeration(m, n, k, tile):
    shape = MatmulShape(m, n, k)
    tiled = TiledMatmulTraffic(shape, tile=tile)

    assert enumerate_tiled_slots(shape, tile) == (
        tiled.tile_slot_loads,
        tiled.useful_a_loads,
        tiled.useful_b_loads,
    )


def test_unit_tile_reproduces_naive_loads():
    shape = MatmulShape(9, 5, 3)

    assert TiledMatmulTraffic(shape, tile=1).useful_loads == (
        NaiveMatmulTraffic(shape).load_elements
    )
    assert TiledMatmulTraffic(shape, tile=1).padded_slots == 0


def test_tile_larger_than_every_dimension_runs_one_block_and_phase():
    tiled = TiledMatmulTraffic(MatmulShape(5, 3, 2), tile=32)

    assert (tiled.blocks, tiled.phases) == (1, 1)
    assert tiled.tile_slot_loads == 2048
    assert (tiled.useful_a_loads, tiled.useful_b_loads) == (10, 6)
    assert tiled.padded_slots == 2032


def test_half_precision_halves_bytes_and_doubles_intensity():
    shape = MatmulShape(128, 128, 128)
    single = TiledMatmulTraffic(shape, tile=32)
    half = TiledMatmulTraffic(shape, tile=32, dtype="bfloat16")

    assert half.shared_memory_bytes_per_block == 4096
    assert half.total_bytes * 2 == single.total_bytes
    assert half.arithmetic_intensity == 2 * single.arithmetic_intensity


def test_traffic_mappings_carry_every_derived_count():
    tiled = TiledMatmulTraffic(MatmulShape(10, 6, 7), tile=4).to_mapping()
    naive = NaiveMatmulTraffic(MatmulShape(10, 6, 7)).to_mapping()

    assert tiled["padded_slots"] == 118
    assert tiled["load_bytes"] == 266 * 4
    assert tiled["flops"] == naive["flops"] == 840
    assert naive["load_elements"] == 840
    assert MatmulShape(10, 6, 7).to_mapping() == {"m": 10, "n": 6, "k": 7}


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: MatmulShape(0, 1, 1), MemoryAccessError, "m must be in"),
        (lambda: MatmulShape(1, 1 << 17, 1), MemoryAccessError, "n must be in"),
        (lambda: MatmulShape(1, 1, 2.0), TypeError, "k must be an integer"),
        (lambda: MatmulShape(True, 1, 1), TypeError, "m must be an integer"),
        (
            lambda: TiledMatmulTraffic(MatmulShape(4, 4, 4), tile=0),
            MemoryAccessError,
            "tile",
        ),
        (
            lambda: TiledMatmulTraffic(MatmulShape(4, 4, 4), tile=33),
            MemoryAccessError,
            "tile",
        ),
        (
            lambda: TiledMatmulTraffic(MatmulShape(4, 4, 4), tile=True),
            TypeError,
            "tile",
        ),
        (
            lambda: NaiveMatmulTraffic(MatmulShape(4, 4, 4), dtype="int8"),
            MemoryAccessError,
            "dtype",
        ),
        (lambda: NaiveMatmulTraffic((4, 4, 4)), TypeError, "MatmulShape"),
    ],
)
def test_tiling_contracts_reject_ambiguous_inputs(call, error, message):
    with pytest.raises(error, match=message):
        call()
