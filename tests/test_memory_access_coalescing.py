import pytest

from ai_journey.memory_access_coalescing import (
    analyze_coalescing,
    summarize_coalescing,
)
from ai_journey.memory_access_pattern import (
    MatrixLayout,
    MemoryAccessError,
    broadcast_access,
    column_access,
    contiguous_access,
    lane_mask,
    permuted_access,
    row_access,
    strided_access,
)


def test_contiguous_float32_warp_uses_four_sectors_in_one_line():
    analysis = analyze_coalescing(contiguous_access())

    assert analysis.sectors == (0, 1, 2, 3)
    assert analysis.lines == (0,)
    assert analysis.requested_bytes == 128
    assert analysis.unique_bytes == 128
    assert analysis.transferred_bytes == 128
    assert analysis.efficiency == 1.0


def test_stride_two_float32_doubles_sectors_and_halves_efficiency():
    analysis = analyze_coalescing(strided_access(stride_elements=2))

    assert analysis.sector_count == 8
    assert analysis.line_count == 2
    assert analysis.transferred_bytes == 256
    assert analysis.efficiency == 0.5


@pytest.mark.parametrize("stride", [8, 9, 16, 32, 1024])
def test_float32_strides_of_eight_or_more_need_one_sector_per_lane(stride):
    analysis = analyze_coalescing(strided_access(stride_elements=stride))

    assert analysis.sector_count == 32
    assert analysis.transferred_bytes == 1024
    assert analysis.efficiency == 0.125


def test_stride_four_float32_uses_a_quarter_of_each_sector():
    analysis = analyze_coalescing(strided_access(stride_elements=4))

    assert (analysis.sector_count, analysis.line_count) == (16, 4)
    assert analysis.efficiency == 0.25


def test_four_byte_base_offset_spills_into_a_fifth_sector_and_second_line():
    analysis = analyze_coalescing(contiguous_access(base_bytes=4))

    assert analysis.sectors == (0, 1, 2, 3, 4)
    assert analysis.lines == (0, 1)
    assert analysis.transferred_bytes == 160
    assert analysis.efficiency == 0.8
    sector_aligned = analyze_coalescing(contiguous_access(base_bytes=32))
    assert sector_aligned.sector_count == 4
    assert sector_aligned.line_count == 2


def test_broadcast_moves_one_sector_for_every_requesting_lane():
    analysis = analyze_coalescing(broadcast_access(address_bytes=64))

    assert analysis.sectors == (2,)
    assert analysis.lines == (0,)
    assert analysis.requested_bytes == 128
    assert analysis.unique_bytes == 4
    assert analysis.efficiency == 0.125


def test_inactive_lanes_request_nothing():
    first_eight = analyze_coalescing(contiguous_access().masked(lane_mask(range(8))))
    edges = analyze_coalescing(contiguous_access().masked(lane_mask((0, 31))))

    assert first_eight.active_lanes == 8
    assert first_eight.sectors == (0,)
    assert first_eight.efficiency == 1.0
    assert edges.sectors == (0, 3)
    assert edges.requested_bytes == 8
    assert edges.efficiency == 0.125


def test_permuting_lanes_within_the_span_keeps_four_sectors():
    analysis = analyze_coalescing(permuted_access(seed=74))

    assert analysis.sectors == (0, 1, 2, 3)
    assert analysis.efficiency == 1.0


@pytest.mark.parametrize(
    ("width", "sectors", "lines"),
    [(1, 1, 1), (2, 2, 1), (4, 4, 1), (8, 8, 2), (16, 16, 4)],
)
def test_contiguous_access_width_scales_sectors_and_lines(width, sectors, lines):
    analysis = analyze_coalescing(contiguous_access(access_bytes=width))

    assert analysis.sector_count == sectors
    assert analysis.line_count == lines
    assert analysis.efficiency == 1.0


def test_padding_the_leading_dimension_to_a_warp_multiple_keeps_rows_aligned():
    unpadded = MatrixLayout(rows=8, columns=33, leading_dimension=33)
    padded = MatrixLayout(rows=8, columns=33, leading_dimension=64)

    unpadded_row = analyze_coalescing(row_access(unpadded, row=1))
    padded_row = analyze_coalescing(row_access(padded, row=1))

    assert unpadded_row.sector_count == 5
    assert padded_row.sector_count == 4
    assert analyze_coalescing(column_access(padded, column=0)).active_lanes == 8


def test_column_walk_touches_one_sector_per_active_lane():
    layout = MatrixLayout(rows=64, columns=64, leading_dimension=64)
    analysis = analyze_coalescing(column_access(layout, column=5))

    assert analysis.sector_count == 32
    assert analysis.line_count == 32
    assert analysis.efficiency == 0.125


def test_summary_adds_independent_requests_without_reuse():
    analyses = (
        analyze_coalescing(contiguous_access()),
        analyze_coalescing(contiguous_access(base_bytes=4)),
        analyze_coalescing(broadcast_access()),
    )
    summary = summarize_coalescing(analyses)

    assert summary.requests == 3
    assert summary.sectors == 10
    assert summary.lines == 4
    assert summary.requested_bytes == 384
    assert summary.unique_bytes == 260
    assert summary.transferred_bytes == 320
    assert summary.sectors_per_request == 10 / 3
    assert summary.efficiency == 260 / 320
    assert summary.scaled(4).sectors == 40
    assert summary.scaled(4).sectors_per_request == summary.sectors_per_request
    mapping = summary.to_mapping()
    assert mapping["transferred_bytes"] == 320
    assert mapping["efficiency"] == summary.efficiency


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: analyze_coalescing("pattern"), TypeError),
        (lambda: summarize_coalescing(()), MemoryAccessError),
        (
            lambda: summarize_coalescing([analyze_coalescing(contiguous_access())]),
            MemoryAccessError,
        ),
        (lambda: summarize_coalescing(("analysis",)), TypeError),
        (
            lambda: summarize_coalescing(
                (analyze_coalescing(contiguous_access()),)
            ).scaled(0),
            MemoryAccessError,
        ),
        (
            lambda: summarize_coalescing(
                (analyze_coalescing(contiguous_access()),)
            ).scaled(True),
            TypeError,
        ),
    ],
)
def test_coalescing_contracts_reject_invalid_inputs(call, error):
    with pytest.raises(error):
        call()


def test_analysis_mapping_reports_derived_byte_accounting():
    mapping = analyze_coalescing(contiguous_access(base_bytes=4)).to_mapping()

    assert mapping["sectors"] == 5
    assert mapping["transferred_bytes"] == 160
    assert mapping["sector_indices"] == [0, 1, 2, 3, 4]
    assert mapping["line_indices"] == [0, 1]
    assert mapping["efficiency"] == 0.8
