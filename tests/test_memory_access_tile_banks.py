import pytest

from ai_journey.memory_access_banks import BANK_EXCLUSION
from ai_journey.memory_access_pattern import MemoryAccessError
from ai_journey.memory_access_tile_banks import (
    INNER_LOOP_LAYOUTS,
    analyze_inner_loop_banks,
    inner_loop_access,
    tile_warp_threads,
    tile_warps,
)


def test_warps_hold_consecutive_thread_ids_with_x_fastest():
    lanes = tile_warp_threads(8, 1)

    assert tile_warps(8) == 2
    assert lanes[0] == (4, 0)
    assert lanes[7] == (4, 7)
    assert lanes[8] == (5, 0)
    assert tile_warp_threads(32, 3) == tuple((3, tx) for tx in range(32))
    assert tile_warp_threads(4, 0)[16:] == (None,) * 16


def test_a_tile_row_reads_are_broadcast_within_each_tile_row():
    access = inner_loop_access(32, warp=5, step=7, layout="a_row_major")
    analysis = analyze_inner_loop_banks(32)

    assert set(access.active_addresses) == {(5 * 32 + 7) * 4}
    assert analysis.example("a_row_major").distinct_words == 1
    assert analysis.example("a_row_major").multicast_words == 1
    assert analysis.summary("a_row_major").max_conflict_degree == 1


def test_b_tile_row_reads_walk_consecutive_banks():
    access = inner_loop_access(32, warp=0, step=3, layout="b_row_major")

    assert access.active_addresses == tuple((3 * 32 + tx) * 4 for tx in range(32))
    assert analyze_inner_loop_banks(32).summary("b_row_major").wavefronts == 1024


@pytest.mark.parametrize(
    ("tile", "requests", "column_major_degree", "column_major_wavefronts"),
    [(8, 16, 2, 32), (16, 128, 8, 1024), (32, 1024, 32, 32_768)],
)
def test_column_major_b_tile_conflicts_until_padded(
    tile, requests, column_major_degree, column_major_wavefronts
):
    analysis = analyze_inner_loop_banks(tile)

    for layout in ("a_row_major", "b_row_major", "b_column_major_padded"):
        summary = analysis.summary(layout)
        assert summary.requests == requests
        assert summary.wavefronts == requests
        assert summary.max_conflict_degree == 1
    column_major = analysis.summary("b_column_major")
    assert column_major.max_conflict_degree == column_major_degree
    assert column_major.wavefronts == column_major_wavefronts


def test_warps_spanning_partial_rows_still_conflict_on_column_major_b():
    analysis = analyze_inner_loop_banks(24)

    assert analysis.warps == 18
    assert analysis.summary("b_column_major").max_conflict_degree == 6
    assert analysis.summary("b_column_major_padded").max_conflict_degree == 1
    assert analysis.example("b_row_major").multicast_words == 8


def test_one_element_of_padding_is_not_enough_for_narrow_elements():
    halves = analyze_inner_loop_banks(16, element_bytes=2)
    bytes_view = analyze_inner_loop_banks(16, element_bytes=1)

    assert halves.summary("b_column_major").max_conflict_degree == 4
    assert halves.summary("b_column_major_padded").max_conflict_degree == 2
    assert bytes_view.summary("b_column_major_padded").max_conflict_degree == 2
    assert halves.summary("b_row_major").max_conflict_degree == 1


def test_small_tiles_leave_lanes_inactive_without_conflicts():
    analysis = analyze_inner_loop_banks(4)

    assert analysis.warps == 1
    assert analysis.example("b_column_major").active_lanes == 16
    assert all(
        analysis.summary(layout).max_conflict_degree == 1
        for layout in INNER_LOOP_LAYOUTS
    )


def test_inner_loop_mapping_reports_every_layout():
    mapping = analyze_inner_loop_banks(16).to_mapping()

    assert (mapping["tile"], mapping["warps"], mapping["steps"]) == (16, 8, 16)
    assert set(mapping["layouts"]) == set(INNER_LOOP_LAYOUTS)
    column_major = mapping["layouts"]["b_column_major"]
    assert column_major["all_steps"]["max_conflict_degree"] == 8
    assert column_major["warp0_step0"]["conflict_degree"] == 8


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: tile_warps(0), MemoryAccessError, "tile"),
        (lambda: tile_warps(33), MemoryAccessError, "tile"),
        (lambda: tile_warp_threads(8, 2), MemoryAccessError, "warp"),
        (lambda: tile_warp_threads(8, True), TypeError, "warp"),
        (
            lambda: inner_loop_access(8, warp=0, step=8, layout="b_row_major"),
            MemoryAccessError,
            "step",
        ),
        (
            lambda: inner_loop_access(8, warp=0, step=0, layout="diagonal"),
            MemoryAccessError,
            "layout",
        ),
        (
            lambda: analyze_inner_loop_banks(8, element_bytes=8),
            MemoryAccessError,
            BANK_EXCLUSION[:20],
        ),
    ],
)
def test_inner_loop_contracts_reject_invalid_inputs(call, error, message):
    with pytest.raises(error, match=message):
        call()
