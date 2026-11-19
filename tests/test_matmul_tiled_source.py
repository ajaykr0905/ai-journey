import re

import pytest

from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.matmul_tiled_source import (
    BARRIER,
    HEADER,
    LOOP_STATEMENT,
    STORE_GUARD,
    TILE_SIZES,
    require_tile,
    threadgroup_bytes,
    tiled_kernel_name,
    tiled_kernel_source,
)


def phase_body(source):
    """Return the text of the phase loop, between its header and its close."""

    start = source.index("    for (uint start = 0;")
    end = source.index(f"    {STORE_GUARD}")
    return source[start:end]


def model_steps(k, tile):
    """Walk the kernel's loop bounds in Python: every k index the fma chain reads."""

    steps = []
    for start in range(0, k, tile):
        for i in range(min(tile, k - start)):
            steps.append(start + i)
    return steps


@pytest.mark.parametrize("tile", TILE_SIZES)
def test_each_kernel_declares_two_static_tiles_of_its_own_size(tile):
    source = tiled_kernel_source(tile)

    assert f"kernel void {tiled_kernel_name(tile)}(" in source
    assert f"    threadgroup float a_tile[{tile}][{tile}];\n" in source
    assert f"    threadgroup float b_tile[{tile}][{tile}];\n" in source
    assert source.count("threadgroup float") == 2
    assert threadgroup_bytes(tile) == 2 * tile * tile * 4
    assert "[[thread_position_in_threadgroup]]" in source
    assert source.isascii() and not source.startswith(HEADER)


@pytest.mark.parametrize("tile", TILE_SIZES)
def test_every_thread_reaches_both_barriers_and_only_the_store_is_guarded(tile):
    source = tiled_kernel_source(tile)
    body = phase_body(source)

    assert body.count(BARRIER) == 2 and source.count(BARRIER) == 2
    assert "if (" not in body and "return" not in source
    assert source.count("if (") == 1
    guarded = source[source.index(STORE_GUARD) :]
    assert guarded.splitlines()[1] == "        c[row * n + column] = sum;"
    assert source.count("c[") == 1


@pytest.mark.parametrize("tile", TILE_SIZES)
def test_out_of_range_slots_are_zero_filled_and_the_k_loop_stops_at_k(tile):
    body = phase_body(tiled_kernel_source(tile))

    assert "row < m && a_column < k ? a[row * k + a_column] : 0.0f;" in body
    assert "b_row < k && column < n ? b[b_row * n + column] : 0.0f;" in body
    assert f"for (uint start = 0; start < k; start += {tile}) {{" in body
    assert f"uint steps = min({tile}u, k - start);" in body
    assert "for (uint i = 0; i < steps; ++i) {" in body
    assert body.count(LOOP_STATEMENT) == 1 and "fma(" in LOOP_STATEMENT
    first_barrier, second_barrier = (
        match.start() for match in re.finditer(re.escape(BARRIER), body)
    )
    assert first_barrier < body.index(LOOP_STATEMENT) < second_barrier


@pytest.mark.parametrize("tile", TILE_SIZES)
def test_the_loop_bounds_take_exactly_k_steps_in_increasing_k(tile):
    for k in (*range(1, 3 * tile + 2), 1000, 4096):
        assert model_steps(k, tile) == list(range(k)), k


def test_a_tile_must_fit_one_threadgroup():
    assert [require_tile(tile) for tile in (1, 8, 32)] == [1, 8, 32]
    for tile in (0, 33):
        with pytest.raises(KernelMetalError, match="tile must be in"):
            require_tile(tile)
    for tile in (True, 16.0, "16"):
        with pytest.raises(TypeError, match="tile must be an integer"):
            tiled_kernel_source(tile)
