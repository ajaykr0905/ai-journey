import pytest

from ai_journey.simt_device import PINNED_ARCHITECTURES, SimtError
from ai_journey.simt_launch import (
    Dim3,
    LaunchConfiguration,
    check_launch,
    launch_for_elements,
    launch_for_image,
)

AMPERE = PINNED_ARCHITECTURES["8.0"]
TURING = PINNED_ARCHITECTURES["7.5"]


def test_vector_add_grid_uses_ceil_division_over_elements():
    # 2^24 / 256 = 65,536 blocks exactly; 256 threads form 8 full warps.
    geometry = launch_for_elements(AMPERE, elements=2**24, threads_per_block=256)

    assert geometry.launch == LaunchConfiguration(Dim3(65_536), Dim3(256))
    assert geometry.warps_per_block == 8
    assert geometry.idle_lanes_per_block == 0
    assert geometry.total_threads == 2**24
    assert geometry.total_warps == 524_288
    assert geometry.lane_utilization == 1.0

    # 1,000 elements need ceil(1000 / 256) = 4 blocks, i.e. 24 surplus threads.
    ragged = launch_for_elements(AMPERE, elements=1_000, threads_per_block=256)
    assert ragged.launch.grid == Dim3(4)
    assert ragged.total_threads == 1_024


def test_image_grid_rounds_rows_and_columns_up_independently():
    # 1920 / 16 = 120 columns of blocks; ceil(1080 / 16) = 68 rows of blocks.
    geometry = launch_for_image(AMPERE, width=1920, height=1080, block_x=16, block_y=16)

    assert geometry.launch == LaunchConfiguration(Dim3(120, 68), Dim3(16, 16))
    assert geometry.total_blocks == 8_160
    assert geometry.total_threads == 2_088_960
    assert geometry.warps_per_block == 8
    assert geometry.to_mapping()["grid"] == [120, 68, 1]
    assert geometry.to_mapping()["block"] == [16, 16, 1]


def test_partial_warps_leave_idle_lanes_in_every_block():
    # 100 threads need ceil(100 / 32) = 4 warps = 128 lanes, so 28 lanes idle.
    flat = check_launch(AMPERE, LaunchConfiguration(Dim3(3), Dim3(100)))
    square = check_launch(AMPERE, LaunchConfiguration(Dim3(3), Dim3(10, 10)))

    for geometry in (flat, square):
        assert geometry.warps_per_block == 4
        assert geometry.idle_lanes_per_block == 28
        assert geometry.total_idle_lanes == 84
        assert geometry.lane_utilization == 100 / 128
    single = check_launch(TURING, LaunchConfiguration(Dim3(1), Dim3(1)))
    assert (single.warps_per_block, single.idle_lanes_per_block) == (1, 31)


def test_largest_legal_dimensions_are_admitted():
    edges = check_launch(
        AMPERE,
        LaunchConfiguration(Dim3(2**31 - 1, 65_535, 65_535), Dim3(1024)),
    )
    deep = check_launch(AMPERE, LaunchConfiguration(Dim3(1), Dim3(16, 1, 64)))

    assert edges.total_blocks == (2**31 - 1) * 65_535 * 65_535
    assert deep.threads_per_block == 1_024


@pytest.mark.parametrize(
    ("launch", "message"),
    [
        (LaunchConfiguration(Dim3(1), Dim3(1025)), "block x dimension 1025"),
        (LaunchConfiguration(Dim3(1), Dim3(1, 1025)), "block y dimension 1025"),
        (LaunchConfiguration(Dim3(1), Dim3(1, 1, 65)), "block z dimension 65"),
        (LaunchConfiguration(Dim3(1), Dim3(32, 32, 2)), "2048 threads"),
        (LaunchConfiguration(Dim3(1, 65_536), Dim3(32)), "grid y dimension 65536"),
        (LaunchConfiguration(Dim3(1, 1, 65_536), Dim3(32)), "grid z dimension"),
    ],
)
def test_launches_beyond_cuda_dimension_limits_are_rejected(launch, message):
    with pytest.raises(SimtError, match=message):
        check_launch(AMPERE, launch)


def test_grid_sizing_rejects_shapes_that_need_unsupported_grids():
    # ceil(65_536 * 16 + 1 rows / 16) = 65,537 block rows exceeds grid y.
    with pytest.raises(SimtError, match="grid y dimension 65537"):
        launch_for_image(AMPERE, width=16, height=1_048_577, block_x=16, block_y=16)
    with pytest.raises(SimtError, match="more blocks than one grid dimension"):
        launch_for_elements(AMPERE, elements=2**31 * 32, threads_per_block=1)
    with pytest.raises(SimtError, match="2048 threads"):
        launch_for_image(AMPERE, width=64, height=64, block_x=64, block_y=32)


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: Dim3(0), SimtError, "dim3.x must be between 1"),
        (lambda: Dim3(1, True), TypeError, "dim3.y must be an integer"),
        (lambda: Dim3(1, 1, 2**31), SimtError, "dim3.z must be between"),
        (lambda: LaunchConfiguration((1, 1, 1), Dim3(1)), TypeError, "Dim3"),
        (
            lambda: launch_for_elements(AMPERE, elements=0, threads_per_block=32),
            SimtError,
            "elements must be between",
        ),
        (
            lambda: launch_for_elements("8.0", elements=1, threads_per_block=32),
            TypeError,
            "SimtArchitecture",
        ),
        (
            lambda: launch_for_image(AMPERE, width=4.0, height=4, block_x=2, block_y=2),
            TypeError,
            "width must be an integer",
        ),
    ],
)
def test_launch_contracts_reject_ambiguous_inputs(call, error, message):
    with pytest.raises(error, match=message):
        call()
