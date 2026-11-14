import pytest

from ai_journey.memory_access_pattern import (
    CUDA_GEOMETRY,
    FULL_WARP_MASK,
    MatrixLayout,
    MemoryAccessError,
    MemoryGeometry,
    WarpAccess,
    broadcast_access,
    column_access,
    contiguous_access,
    lane_mask,
    permuted_access,
    require_aligned_address,
    row_access,
    seeded_permutation,
    strided_access,
)


def test_geometry_is_the_documented_cuda_contract():
    assert CUDA_GEOMETRY.to_mapping() == {
        "warp_size": 32,
        "sector_bytes": 32,
        "line_bytes": 128,
        "shared_memory_banks": 32,
        "bank_width_bytes": 4,
    }
    with pytest.raises(MemoryAccessError, match="documented CUDA contract"):
        MemoryGeometry(warp_size=64)
    with pytest.raises(MemoryAccessError, match="documented CUDA contract"):
        MemoryGeometry(bank_width_bytes=8)
    with pytest.raises(TypeError, match="sector_bytes must be an integer"):
        MemoryGeometry(sector_bytes=32.0)


def test_contiguous_strided_and_offset_generators_emit_lane_byte_addresses():
    contiguous = contiguous_access()
    assert contiguous.lane_addresses == tuple(range(0, 128, 4))
    assert contiguous.active_mask == FULL_WARP_MASK
    assert strided_access(stride_elements=2).lane_addresses[:3] == (0, 8, 16)
    assert strided_access(
        stride_elements=3, access_bytes=8, base_bytes=64
    ).lane_addresses[:3] == (64, 88, 112)
    offset = contiguous_access(base_bytes=4)
    assert offset.lane_addresses[0] == 4
    assert offset.lane_addresses[-1] == 128
    assert contiguous_access(access_bytes=1).lane_addresses[-1] == 31


def test_broadcast_requests_one_element_from_every_lane():
    access = broadcast_access(address_bytes=96, access_bytes=4)

    assert set(access.lane_addresses) == {96}
    assert access.active_addresses == (96,) * 32


def test_seeded_permutation_is_pinned_portable_and_complete():
    order = seeded_permutation(74)

    assert sorted(order) == list(range(32))
    assert order[:8] == (17, 28, 26, 9, 31, 27, 11, 20)
    assert seeded_permutation(74) == order
    assert seeded_permutation(75) != order
    assert seeded_permutation(0, 8) == (5, 4, 6, 7, 0, 3, 1, 2)
    access = permuted_access(seed=74, base_bytes=256)
    assert sorted(access.active_addresses) == list(range(256, 384, 4))
    with pytest.raises(MemoryAccessError, match="seed"):
        seeded_permutation(-1)


def test_row_walk_respects_leading_dimension_padding_and_row_end():
    layout = MatrixLayout(rows=4, columns=40, leading_dimension=48)
    access = row_access(layout, row=1, start_column=16)

    assert layout.padding_elements == 8
    assert layout.allocation_bytes == 4 * 48 * 4
    assert access.lane_addresses[:3] == (256, 260, 264)
    assert access.active_lanes == tuple(range(24))
    assert access.lane_addresses[24:] == (None,) * 8


def test_column_walk_strides_by_the_leading_dimension_and_stops_at_last_row():
    layout = MatrixLayout(rows=20, columns=8, leading_dimension=9, base_bytes=128)
    access = column_access(layout, column=3)

    assert access.lane_addresses[:3] == (140, 176, 212)
    assert access.active_lanes == tuple(range(20))
    assert column_access(layout, column=0, start_row=19).active_lanes == (0,)
    with pytest.raises(MemoryAccessError, match="column must be in"):
        column_access(layout, column=8)


def test_active_lane_masks_exclude_lanes_without_inventing_addresses():
    half = contiguous_access().masked(lane_mask(range(16)))

    assert half.active_lanes == tuple(range(16))
    assert half.lane_addresses[16:] == (None,) * 16
    assert half.masked(lane_mask((0, 20))).active_lanes == (0,)
    assert lane_mask((0, 2)) == 5
    assert half.to_mapping()["active_mask"] == 0xFFFF
    with pytest.raises(MemoryAccessError, match="active_mask"):
        contiguous_access().masked(0)
    with pytest.raises(MemoryAccessError, match="active_mask"):
        contiguous_access().masked(1 << 32)
    with pytest.raises(TypeError, match="active_mask"):
        contiguous_access().masked(True)
    with pytest.raises(MemoryAccessError, match="unique"):
        lane_mask((3, 3))
    with pytest.raises(MemoryAccessError, match="lane must be in"):
        lane_mask((32,))


@pytest.mark.parametrize("width", [1, 2, 4, 8, 16])
def test_natural_alignment_accepts_multiples_of_each_supported_width(width):
    assert require_aligned_address(width * 3, width) == width * 3
    if width > 1:
        with pytest.raises(MemoryAccessError, match="naturally aligned"):
            require_aligned_address(width * 3 + 1, width)


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: require_aligned_address(0, 3), MemoryAccessError, "1, 2, 4, 8"),
        (lambda: require_aligned_address(True, 4), TypeError, "address"),
        (lambda: require_aligned_address(-4, 4), MemoryAccessError, "byte range"),
        (lambda: contiguous_access(base_bytes=2), MemoryAccessError, "aligned"),
        (lambda: strided_access(stride_elements=0), MemoryAccessError, "stride"),
        (lambda: WarpAccess(4, (0,) * 31), MemoryAccessError, "one entry per lane"),
        (lambda: WarpAccess(4, (None,) * 32), MemoryAccessError, "active lane"),
        (lambda: WarpAccess(4, [0] * 32), MemoryAccessError, "one entry per lane"),
        (
            lambda: MatrixLayout(rows=2, columns=8, leading_dimension=7),
            MemoryAccessError,
            "leading_dimension",
        ),
        (
            lambda: MatrixLayout(rows=2, columns=8, leading_dimension=8, base_bytes=6),
            MemoryAccessError,
            "aligned",
        ),
    ],
)
def test_pattern_contracts_reject_ambiguous_inputs(call, error, message):
    with pytest.raises(error, match=message):
        call()
