import itertools

import pytest

from ai_journey.simt_device import PINNED_ARCHITECTURES, SimtError
from ai_journey.simt_indexing import (
    coverage_1d,
    coverage_2d,
    global_index_1d,
    global_index_2d,
)
from ai_journey.simt_launch import (
    Dim3,
    LaunchConfiguration,
    check_launch,
    launch_for_elements,
    launch_for_image,
)

AMPERE = PINNED_ARCHITECTURES["8.0"]


def admitted(grid, block):
    return check_launch(AMPERE, LaunchConfiguration(Dim3(*grid), Dim3(*block)))


def enumerate_threads(geometry, width, height):
    """Visit every thread in thread-ID order and tally guard outcomes."""

    launch = geometry.launch
    covered = set()
    masked = out_of_bounds = aliased = 0
    warps = {"active": 0, "partial": 0, "masked": 0}
    for block_y, block_x in itertools.product(
        range(launch.grid.y), range(launch.grid.x)
    ):
        passing = []
        for thread_y, thread_x in itertools.product(
            range(launch.block.y), range(launch.block.x)
        ):
            column = block_x * launch.block.x + thread_x
            row = block_y * launch.block.y + thread_y
            linear = row * width + column
            inside = column < width and row < height
            passing.append(inside)
            if inside:
                covered.add(linear)
            else:
                masked += 1
                if linear >= width * height:
                    out_of_bounds += 1
                else:
                    aliased += 1
        for first in range(0, len(passing), 32):
            lanes = passing[first : first + 32]
            if all(lanes):
                warps["active"] += 1
            elif any(lanes):
                warps["partial"] += 1
            else:
                warps["masked"] += 1
    return {
        "launched_threads": launch.total_threads,
        "covered_elements": len(covered),
        "uncovered_elements": width * height - len(covered),
        "masked_threads": masked,
        "unguarded_out_of_bounds_writes": out_of_bounds,
        "unguarded_aliased_writes": aliased,
        "fully_active_warps": warps["active"],
        "partially_masked_warps": warps["partial"],
        "fully_masked_warps": warps["masked"],
    }


COUNT_FIELDS = (
    "launched_threads",
    "covered_elements",
    "uncovered_elements",
    "masked_threads",
    "unguarded_out_of_bounds_writes",
    "unguarded_aliased_writes",
    "fully_active_warps",
    "partially_masked_warps",
    "fully_masked_warps",
)


def counts(coverage):
    mapping = coverage.to_mapping()
    return {key: mapping[key] for key in COUNT_FIELDS}


def test_global_index_helpers_follow_the_cuda_formulas():
    assert global_index_1d(block_index=3, block_dim=256, thread_index=7) == 775
    # Block (2, 1) of 16x16 threads, thread (5, 9): column 37, row 25.
    assert global_index_2d(
        block_index=(2, 1), block_dim=(16, 16), thread_index=(5, 9), width=1920
    ) == (25, 37, 25 * 1920 + 37)
    with pytest.raises(SimtError, match="thread_index must be between 0 and 255"):
        global_index_1d(block_index=0, block_dim=256, thread_index=256)
    with pytest.raises(SimtError, match="must be an \\(x, y\\) tuple"):
        global_index_2d(
            block_index=[0, 0], block_dim=(1, 1), thread_index=(0, 0), width=1
        )


def test_vector_add_launch_covers_every_element_without_masked_lanes():
    geometry = launch_for_elements(AMPERE, elements=2**24, threads_per_block=256)
    coverage = coverage_1d(geometry, elements=2**24)

    assert coverage.covered_elements == 2**24
    assert coverage.masked_threads == 0
    assert coverage.under_covered is False
    assert coverage.needs_bounds_guard is False
    assert coverage.fully_active_warps == 2**24 // 32
    assert coverage.partially_masked_warps == coverage.fully_masked_warps == 0


def test_ragged_1d_launch_masks_the_tail_of_its_last_warp():
    # 4 blocks * 256 = 1,024 threads for 1,000 elements: threads 1000..1023 fail
    # the guard. Warp 31 holds threads 992..1023, so 8 lanes pass and 24 do not.
    geometry = launch_for_elements(AMPERE, elements=1_000, threads_per_block=256)
    coverage = coverage_1d(geometry, elements=1_000)

    assert coverage.masked_threads == 24
    assert coverage.unguarded_out_of_bounds_writes == 24
    assert coverage.unguarded_aliased_writes == 0
    assert coverage.fully_active_warps == 31
    assert coverage.partially_masked_warps == 1
    assert coverage.fully_masked_warps == 0
    assert coverage.needs_bounds_guard is True


def test_grayscale_launch_masks_only_whole_warps_in_the_last_block_row():
    # ceil(1080 / 16) = 68 block rows cover 1,088 rows; rows 1080..1087 are
    # 8 * 1920 = 15,360 surplus threads. A 16x16 block packs two rows per warp,
    # so warps 4..7 of each of the 120 bottom blocks are entirely masked.
    geometry = launch_for_image(AMPERE, width=1920, height=1080, block_x=16, block_y=16)
    coverage = coverage_2d(geometry, width=1920, height=1080)

    assert coverage.covered_elements == 2_073_600
    assert coverage.masked_threads == 15_360
    assert coverage.unguarded_out_of_bounds_writes == 15_360
    assert coverage.unguarded_aliased_writes == 0
    assert coverage.fully_masked_warps == 120 * 4
    assert coverage.partially_masked_warps == 0
    assert coverage.fully_active_warps == 8_160 * 8 - 480


def test_floor_division_grid_under_covers_the_image_bottom():
    # 1080 // 16 = 67 block rows reach row 1071, leaving 8 rows uncovered.
    coverage = coverage_2d(admitted((120, 67), (16, 16)), width=1920, height=1080)

    assert coverage.under_covered is True
    assert coverage.uncovered_elements == 15_360
    assert coverage.masked_threads == 0


def test_unguarded_2d_threads_past_a_row_end_alias_the_next_row():
    # Width 5, height 3, blocks of 4x2 on a 2x2 grid: 8 x 4 = 32 threads.
    # Covered: 5 * 3 = 15. Masked: 32 - 15 = 17.
    # Row 3 (below the image): 8 threads, all beyond index 15.
    # Rows 0..2, columns 5..7: linear r*5 + c is 5..7, 10..12, 15..17, so the
    # 3 threads of row 2 overflow and the 6 of rows 0 and 1 alias row starts.
    coverage = coverage_2d(admitted((2, 2), (4, 2)), width=5, height=3)

    assert coverage.masked_threads == 17
    assert coverage.unguarded_out_of_bounds_writes == 8 + 3
    assert coverage.unguarded_aliased_writes == 6


def test_closed_form_counts_match_brute_force_enumeration():
    cases = 0
    for (width, height), (block_x, block_y) in itertools.product(
        ((1, 1), (5, 3), (17, 1), (17, 9), (33, 3), (6, 9)),
        ((4, 3), (5, 2), (16, 2), (8, 8), (3, 7), (32, 1)),
    ):
        ceil_x = -(-width // block_x)
        ceil_y = -(-height // block_y)
        for grid_x, grid_y in itertools.product(
            {max(1, ceil_x - 1), ceil_x, ceil_x + 1},
            {max(1, ceil_y - 1), ceil_y, ceil_y + 2},
        ):
            geometry = admitted((grid_x, grid_y), (block_x, block_y))
            expected = enumerate_threads(geometry, width, height)
            actual = coverage_2d(geometry, width=width, height=height)
            assert counts(actual) == expected, (width, height, geometry.launch)
            cases += 1
    for elements, block in itertools.product((1, 31, 33, 100, 257), (32, 48, 100)):
        ceil_blocks = -(-elements // block)
        for blocks in {max(1, ceil_blocks - 1), ceil_blocks, ceil_blocks + 2}:
            geometry = admitted((blocks,), (block,))
            expected = enumerate_threads(geometry, elements, 1)
            assert counts(coverage_1d(geometry, elements=elements)) == expected
            cases += 1
    assert cases == 269


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda: coverage_1d(admitted((2, 2), (32,)), elements=10),
            SimtError,
            "one-dimensional grid and block",
        ),
        (
            lambda: coverage_2d(admitted((2, 2, 2), (4, 4)), width=4, height=4),
            SimtError,
            "z extents of one",
        ),
        (
            lambda: coverage_1d(LaunchConfiguration(Dim3(1), Dim3(32)), elements=1),
            TypeError,
            "admitted by check_launch",
        ),
        (
            lambda: coverage_2d(admitted((1,), (32,)), width=0, height=1),
            SimtError,
            "width must be between",
        ),
        (
            lambda: coverage_1d(admitted((1,), (32,)), elements=True),
            TypeError,
            "elements must be an integer",
        ),
    ],
)
def test_coverage_rejects_mismatched_layouts_and_invalid_shapes(call, error, message):
    with pytest.raises(error, match=message):
        call()
