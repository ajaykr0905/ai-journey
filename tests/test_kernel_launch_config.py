import itertools

import pytest

from ai_journey.kernel_launch_config import (
    COMPUTE_CAPABILITY,
    LAUNCH_LIMITS,
    LIMITS_SOURCE,
    MAX_BLOCK_DIM_X,
    MAX_BLOCK_DIM_Y,
    MAX_BLOCK_DIM_Z,
    MAX_GRID_DIM_X,
    MAX_GRID_DIM_Y,
    MAX_GRID_DIM_Z,
    MAX_THREADS_PER_BLOCK,
    UINT_MAX,
    WARP_SIZE,
    Dim3,
    Index3,
    KernelLaunchError,
    LaunchConfig,
    LaunchLimitError,
    ceil_div,
    grid_for,
    launch_1d,
    launch_2d,
)
from ai_journey.simt_device import PINNED_ARCHITECTURES


def test_named_limits_match_the_pinned_compute_capability_8_0_record():
    # The SIMT lab pins the same Table 21 row from its own cited source.
    ampere = PINNED_ARCHITECTURES[COMPUTE_CAPABILITY]

    assert COMPUTE_CAPABILITY == "8.0"
    assert (WARP_SIZE, MAX_THREADS_PER_BLOCK) == (32, 1024)
    assert (MAX_BLOCK_DIM_X, MAX_BLOCK_DIM_Y, MAX_BLOCK_DIM_Z) == (1024, 1024, 64)
    assert (MAX_GRID_DIM_X, MAX_GRID_DIM_Y, MAX_GRID_DIM_Z) == (2**31 - 1, 65535, 65535)
    assert dict(LAUNCH_LIMITS) == {
        name: getattr(ampere, name)
        for name in (
            "warp_size",
            "max_threads_per_block",
            "max_block_dim_x",
            "max_block_dim_y",
            "max_block_dim_z",
            "max_grid_dim_x",
            "max_grid_dim_y",
            "max_grid_dim_z",
        )
    }
    assert "Table 21" in LIMITS_SOURCE
    assert "Table 21" in ampere.sources[0]


def test_dim3_defaults_unset_axes_to_one_and_index3_to_zero():
    assert Dim3(7) == Dim3(7, 1, 1)
    assert Dim3(7).volume == 7
    assert Dim3(2, 3, 4).to_list() == [2, 3, 4]
    assert Index3(5) == Index3(5, 0, 0)
    assert Dim3.from_list([16, 16, 1], "block") == Dim3(16, 16)


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: Dim3(0), KernelLaunchError, "dim3.x must be between 1"),
        (lambda: Dim3(1, True), TypeError, "dim3.y must be an integer"),
        (lambda: Dim3(1, 1, 2.0), TypeError, "dim3.z must be an integer"),
        (lambda: Dim3(UINT_MAX + 1), KernelLaunchError, "dim3.x must be between"),
        (lambda: Index3(-1), KernelLaunchError, "index.x must be between 0"),
        (lambda: Index3(0, False), TypeError, "index.y must be an integer"),
        (lambda: Dim3.from_list([16, 16], "block"), KernelLaunchError, "[x, y, z]"),
        (lambda: Dim3.from_list((1, 1, 1), "grid"), KernelLaunchError, "JSON array"),
        (lambda: LaunchConfig((1, 1, 1), Dim3(1)), TypeError, "Dim3"),
    ],
)
def test_value_types_reject_ambiguous_components(call, error, message):
    with pytest.raises(error, match=message):
        call()


def test_largest_legal_launch_is_admitted():
    edges = LaunchConfig(Dim3(2**31 - 1, 65_535, 65_535), Dim3(1024))
    deep = LaunchConfig(Dim3(1), Dim3(16, 1, 64))
    square = LaunchConfig(Dim3(1), Dim3(32, 32))

    assert edges.blocks == (2**31 - 1) * 65_535 * 65_535
    assert edges.launched_threads == edges.blocks * 1024
    assert deep.threads_per_block == square.threads_per_block == 1024


@pytest.mark.parametrize(
    ("grid", "block", "limit", "subject"),
    [
        (Dim3(1), Dim3(1025), "max_block_dim_x", "block.x = 1025"),
        (Dim3(1), Dim3(1, 1025), "max_block_dim_y", "block.y = 1025"),
        (Dim3(1), Dim3(1, 1, 65), "max_block_dim_z", "block.z = 65"),
        (Dim3(1), Dim3(32, 32, 2), "max_threads_per_block", "= 2048"),
        (Dim3(2**31), Dim3(32), "max_grid_dim_x", "grid.x = 2147483648"),
        (Dim3(1, 65_536), Dim3(32), "max_grid_dim_y", "grid.y = 65536"),
        (Dim3(1, 1, 65_536), Dim3(32), "max_grid_dim_z", "grid.z = 65536"),
    ],
)
def test_each_violation_names_the_limit_it_breaks(grid, block, limit, subject):
    with pytest.raises(LaunchLimitError) as raised:
        LaunchConfig(grid, block)

    assert raised.value.limit == limit
    assert raised.value.maximum == LAUNCH_LIMITS[limit]
    message = str(raised.value)
    assert subject in message
    assert f"breaks {limit} = {LAUNCH_LIMITS[limit]}" in message
    assert "compute capability 8.0" in message


def test_block_limits_are_checked_before_the_grid():
    # Both the block and the grid break a limit; the block is reported.
    with pytest.raises(LaunchLimitError) as raised:
        LaunchConfig(Dim3(1, 65_536), Dim3(2048))
    assert raised.value.limit == "max_block_dim_x"


def test_ceil_div_matches_the_lecture_cdiv_formula():
    for numerator, denominator in itertools.product(range(0, 70), range(1, 20)):
        expected = (numerator + denominator - 1) // denominator
        assert ceil_div(numerator, denominator) == expected
        assert ceil_div(numerator, denominator) * denominator >= numerator
    assert ceil_div(2**64, 1) == 2**64
    with pytest.raises(KernelLaunchError, match="denominator must be between 1"):
        ceil_div(10, 0)
    with pytest.raises(KernelLaunchError, match="numerator must be between 0"):
        ceil_div(-1, 4)
    with pytest.raises(TypeError, match="numerator must be an integer"):
        ceil_div(10.0, 4)


def test_one_dimensional_grids_cover_ragged_and_aligned_extents():
    ragged = launch_1d(1000, 256)
    aligned = launch_1d(1024, 256)
    spill = launch_1d(1025, 256)

    assert ragged == LaunchConfig(Dim3(4), Dim3(256))
    assert (ragged.launched_threads, ragged.threads_inside(Dim3(1000))) == (1024, 1000)
    assert aligned.grid == Dim3(4)
    assert aligned.threads_inside(Dim3(1024)) == aligned.launched_threads
    assert spill.grid == Dim3(5)
    assert spill.launched_threads - spill.threads_inside(Dim3(1025)) == 255


def test_two_dimensional_grids_round_columns_and_rows_up_independently():
    image = launch_2d(40, 30, 16, 16)

    assert image == LaunchConfig(Dim3(3, 2), Dim3(16, 16))
    assert image.launched_threads == 48 * 32
    assert image.threads_inside(Dim3(40, 30)) == 1200
    assert grid_for(Dim3(48, 32), Dim3(16, 16)).grid == Dim3(3, 2)
    assert grid_for(Dim3(5, 6, 7), Dim3(2, 4, 8)).grid == Dim3(3, 2, 1)


def test_grid_sizing_names_the_grid_limit_an_extent_needs_to_break():
    with pytest.raises(LaunchLimitError) as raised:
        launch_2d(16, 65_535 * 16 + 1, 16, 16)
    assert raised.value.limit == "max_grid_dim_y"
    assert raised.value.value == 65_536

    with pytest.raises(LaunchLimitError) as raised:
        launch_1d(UINT_MAX, 1)
    assert raised.value.limit == "max_grid_dim_x"

    with pytest.raises(LaunchLimitError) as raised:
        launch_2d(64, 64, 64, 32)
    assert raised.value.limit == "max_threads_per_block"


def test_warps_and_idle_lanes_follow_thread_count():
    # 100 threads need ceil(100 / 32) = 4 warps, so 28 lanes hold no thread.
    for block in (Dim3(100), Dim3(10, 10)):
        config = LaunchConfig(Dim3(3), block)
        assert (config.warps_per_block, config.idle_lanes_per_block) == (4, 28)
    full = launch_2d(40, 30, 16, 16)
    assert full.geometry() == {
        "grid": [3, 2, 1],
        "block": [16, 16, 1],
        "threads_per_block": 256,
        "blocks": 6,
        "launched_threads": 1536,
        "warps_per_block": 8,
        "idle_lanes_per_block": 0,
    }


def test_linear_thread_ids_number_blocks_and_threads_x_fastest():
    config = LaunchConfig(Dim3(3, 2, 2), Dim3(4, 2, 3))
    seen = []
    for bz, by, bx in itertools.product(range(2), range(2), range(3)):
        for tz, ty, tx in itertools.product(range(3), range(2), range(4)):
            block, thread = Index3(bx, by, bz), Index3(tx, ty, tz)
            thread_id = config.thread_id(block, thread)
            assert config.coordinates(thread_id) == (block, thread)
            seen.append(thread_id)

    assert seen == list(range(config.launched_threads))
    with pytest.raises(KernelLaunchError, match="thread index y = 2"):
        config.thread_id(Index3(0), Index3(0, 2))
    with pytest.raises(KernelLaunchError, match="block index x = 3"):
        config.thread_id(Index3(3), Index3(0))
    with pytest.raises(KernelLaunchError, match="thread_id must be between 0"):
        config.coordinates(config.launched_threads)


def test_mappings_round_trip_and_reject_extra_fields():
    config = launch_2d(40, 30, 16, 16)

    assert LaunchConfig.from_mapping(config.to_mapping()) == config
    with pytest.raises(KernelLaunchError, match="exactly grid and block"):
        LaunchConfig.from_mapping({**config.to_mapping(), "shared": 0})
    with pytest.raises(TypeError, match="dim3.x must be an integer"):
        LaunchConfig.from_mapping({"grid": [True, 1, 1], "block": [1, 1, 1]})
    with pytest.raises(LaunchLimitError):
        LaunchConfig.from_mapping({"grid": [1, 1, 1], "block": [1, 1, 128]})
