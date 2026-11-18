import pytest

from ai_journey.kernel_launch_config import Index3, LaunchLimitError
from ai_journey.matmul_model_access import (
    ACCESS_CLASSES,
    MatmulLaunch,
    WarpRef,
    access_class,
    analyze_launch,
    lane_elements,
    lane_stride_bytes,
    line_indices,
    request_mapping,
    sector_indices,
    warp_of_element,
    warp_refs,
    warp_requests,
)
from ai_journey.matmul_model_kernels import X_COLUMNS, X_ROWS
from ai_journey.matmul_model_reference import MatmulModelError
from ai_journey.memory_access_pattern import (
    WarpAccess,
    broadcast_access,
    contiguous_access,
    permuted_access,
    strided_access,
)
from ai_journey.memory_access_tiling import MatmulShape

RAGGED = MatmulShape(m=37, n=45, k=29)
ALIGNED = MatmulShape(m=32, n=64, k=16)


def launch(shape, mapping, block=(32, 8)):
    return MatmulLaunch(shape, block, mapping)


def test_a_warp_is_32_consecutive_x_threads_of_one_block_row():
    columns = launch(ALIGNED, X_COLUMNS)
    rows = launch(ALIGNED, X_ROWS)
    ref = WarpRef(Index3(1, 2), 3)

    # Block (1, 2), warp 3 holds threadIdx.y = 3 and threadIdx.x = 0..31.
    assert lane_elements(columns, ref) == tuple((19, 32 + lane) for lane in range(32))
    assert lane_elements(rows, WarpRef(Index3(0, 2), 3)) == tuple(
        (lane, 19) for lane in range(32)
    )
    wide = launch(MatmulShape(m=4, n=128, k=1), X_COLUMNS, block=(64, 2))
    assert lane_elements(wide, WarpRef(Index3(1, 0), 3)) == tuple(
        (1, 96 + lane) for lane in range(32)
    )
    for current in (columns, rows, wide):
        refs = warp_refs(current)
        assert len(refs) == current.config.blocks * current.warps_per_block
        for ref in refs:
            for row, col in lane_elements(current, ref):
                if row < current.shape.m and col < current.shape.n:
                    assert warp_of_element(current, row, col) == ref


def test_x_over_columns_broadcasts_a_and_coalesces_b_and_c():
    requests = warp_requests(launch(ALIGNED, X_COLUMNS), WarpRef(Index3(0, 0), 0))

    assert requests.active_lanes == 32
    for i, (a_load, b_load) in enumerate(zip(requests.a_loads, requests.b_loads)):
        assert a_load.lane_addresses == (i * 4,) * 32
        assert b_load.lane_addresses == tuple(range(i * 256, i * 256 + 128, 4))
        assert (access_class(a_load), access_class(b_load)) == (
            "broadcast",
            "coalesced",
        )
        assert request_mapping(a_load) | {"lane_addresses": None} == {
            "lane_addresses": None,
            "active_lanes": 32,
            "distinct_addresses": 1,
            "sectors": 1,
            "lines": 1,
            "class": "broadcast",
            "lane_stride_bytes": 0,
        }
        assert (len(sector_indices(b_load)), len(line_indices(b_load))) == (4, 1)
    assert len(requests.a_loads) == len(requests.b_loads) == 16
    assert requests.c_store.lane_addresses == tuple(range(0, 128, 4))


def test_x_over_rows_strides_a_and_c_and_broadcasts_b():
    requests = warp_requests(launch(ALIGNED, X_ROWS), WarpRef(Index3(0, 0), 0))

    for i, (a_load, b_load) in enumerate(zip(requests.a_loads, requests.b_loads)):
        # A rows are 16 floats, 64 bytes, apart: one sector per lane, two
        # lanes per line.
        assert a_load.lane_addresses == tuple((lane * 16 + i) * 4 for lane in range(32))
        assert lane_stride_bytes(a_load) == 64
        assert (len(sector_indices(a_load)), len(line_indices(a_load))) == (32, 16)
        assert b_load.lane_addresses == (i * 256,) * 32
        assert (access_class(a_load), access_class(b_load)) == ("strided", "broadcast")
    store = requests.c_store
    assert (access_class(store), lane_stride_bytes(store)) == ("strided", 256)
    assert (len(sector_indices(store)), len(line_indices(store))) == (32, 32)


def test_a_ragged_edge_warp_keeps_only_the_lanes_inside_c():
    columns = launch(RAGGED, X_COLUMNS)
    edge = warp_of_element(columns, 36, 44)
    requests = warp_requests(columns, edge)

    # Row 36 of columns 32..63: 13 lanes pass the guard.
    assert edge == WarpRef(Index3(1, 4), 4)
    assert requests.active_lanes == 13
    for i, b_load in enumerate(requests.b_loads):
        assert b_load.lane_addresses == tuple(
            (i * 45 + 32 + lane) * 4 if lane < 13 else None for lane in range(32)
        )
        assert access_class(b_load) == "coalesced"
        first, last = (i * 45 + 32) * 4, (i * 45 + 44) * 4
        assert len(sector_indices(b_load)) == last // 32 - first // 32 + 1
    # Rows 37 to 39 of the last block row have no lane inside C.
    assert warp_requests(columns, WarpRef(Index3(0, 4), 5)) is None
    rows = launch(RAGGED, X_ROWS)
    assert warp_requests(rows, warp_of_element(rows, 36, 44)).active_lanes == 5


@pytest.mark.parametrize(
    ("access", "expected", "stride"),
    [
        (broadcast_access(address_bytes=64), "broadcast", 0),
        (contiguous_access(base_bytes=12), "coalesced", 4),
        (contiguous_access().masked(0b1011), "coalesced", 4),
        (strided_access(stride_elements=2), "strided", 8),
        (WarpAccess(4, tuple(4 * (31 - lane) for lane in range(32))), "strided", -4),
        (permuted_access(seed=7), "strided", None),
        (WarpAccess(4, (8,) + (None,) * 31), "single", None),
        (WarpAccess(4, (None, 8, None, 8) + (None,) * 28), "broadcast", 0),
        (WarpAccess(4, (0, None, 4) + (None,) * 29), "strided", 2),
        (WarpAccess(4, (0, 4, 12) + (None,) * 29), "strided", None),
    ],
)
def test_requests_are_classified_by_their_lane_addresses(access, expected, stride):
    assert access_class(access) == expected
    assert lane_stride_bytes(access) == stride
    assert expected in ACCESS_CLASSES


def test_a_request_with_one_active_lane_is_neither_a_broadcast_nor_a_walk():
    # n = 33 leaves only column 32 in the second warp of each row. One lane
    # fits the broadcast rule and the coalesced rule alike, and a C store
    # from one lane is not a broadcast, so the request gets its own class.
    current = launch(MatmulShape(m=4, n=33, k=2), X_COLUMNS, block=(32, 1))
    lone = warp_requests(current, warp_of_element(current, 0, 32))

    assert lone.active_lanes == 1
    for access in (*lone.a_loads, *lone.b_loads, lone.c_store):
        assert access_class(access) == "single"
        assert request_mapping(access)["sectors"] == 1
    classes = {
        name: dict(traffic.classes)
        for name, traffic in analyze_launch(current).operands
    }
    assert classes == {
        "a": {"single": 8, "broadcast": 8, "coalesced": 0, "strided": 0},
        "b": {"single": 8, "broadcast": 0, "coalesced": 8, "strided": 0},
        "c": {"single": 4, "broadcast": 0, "coalesced": 4, "strided": 0},
    }


def test_aligned_launch_totals_match_closed_forms():
    columns = analyze_launch(launch(ALIGNED, X_COLUMNS)).to_mapping()
    rows = analyze_launch(launch(ALIGNED, X_ROWS)).to_mapping()
    steps = 32 * 64 // 32 * 16

    assert columns["warps"] == {"launched": 64, "full": 64, "partial": 0, "idle": 0}
    assert columns["operands"]["a"] == {
        "requests": steps,
        "sectors": steps,
        "lines": steps,
        "requested_bytes": steps * 128,
        "unique_bytes": steps * 4,
        "transferred_bytes": steps * 32,
        "classes": {"single": 0, "broadcast": steps, "coalesced": 0, "strided": 0},
    }
    assert columns["operands"]["b"]["sectors"] == 4 * steps
    assert columns["operands"]["b"]["classes"]["coalesced"] == steps
    assert columns["operands"]["c"]["sectors"] == 4 * 64
    assert rows["operands"]["a"]["sectors"] == 32 * steps
    assert rows["operands"]["a"]["lines"] == 16 * steps
    assert rows["operands"]["b"]["classes"]["broadcast"] == steps
    assert rows["operands"]["c"]["sectors"] == 32 * 64
    # The same useful bytes move 20/3 times the sector bytes when x runs over rows.
    assert (columns["transferred_bytes"], rows["transferred_bytes"]) == (
        steps * 32 + steps * 128 + 64 * 128,
        steps * 32 * 32 + steps * 32 + 64 * 32 * 32,
    )


def test_ragged_launches_count_full_partial_and_idle_warps():
    columns = analyze_launch(launch(RAGGED, X_COLUMNS))
    rows = analyze_launch(launch(RAGGED, X_ROWS))

    assert (columns.warps, columns.full_warps, columns.partial_warps) == (80, 37, 37)
    assert (rows.warps, rows.full_warps, rows.partial_warps) == (96, 45, 45)
    assert columns.idle_warps == rows.idle_warps == 6
    for access in (columns, rows):
        active = access.full_warps + access.partial_warps
        operands = dict(access.operands)
        assert operands["a"].requests == operands["b"].requests == active * 29
        assert operands["c"].requests == active
        assert operands["a"].requested_bytes == 4 * 37 * 45 * 29


def test_the_warp_model_rejects_launches_it_cannot_describe():
    with pytest.raises(MatmulModelError, match="multiple of 32"):
        MatmulLaunch(ALIGNED, (16, 16), X_COLUMNS)
    with pytest.raises(LaunchLimitError, match="max_threads_per_block"):
        MatmulLaunch(ALIGNED, (64, 32), X_COLUMNS)
    with pytest.raises(MatmulModelError, match="mapping must be one of"):
        MatmulLaunch(ALIGNED, (32, 8), "y_columns")
    with pytest.raises(TypeError, match="MatmulShape"):
        MatmulLaunch((32, 64, 16), (32, 8), X_COLUMNS)
    current = launch(ALIGNED, X_COLUMNS)
    with pytest.raises(MatmulModelError, match="outside the launch"):
        lane_elements(current, WarpRef(Index3(2, 0), 0))
    with pytest.raises(MatmulModelError, match="outside the launch"):
        lane_elements(current, WarpRef(Index3(0, 0), 8))
    with pytest.raises(MatmulModelError, match="outside C"):
        warp_of_element(current, 32, 0)
    with pytest.raises(TypeError, match="MatmulLaunch"):
        warp_refs(ALIGNED)
