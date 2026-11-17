from dataclasses import replace

import pytest

from ai_journey import matmul_model_crosscheck
from ai_journey.kernel_launch_config import Index3
from ai_journey.matmul_model_access import MatmulLaunch, WarpRef, warp_requests
from ai_journey.matmul_model_crosscheck import (
    cross_check_launch,
    memory_access_requests,
    simt_warp_classes,
)
from ai_journey.matmul_model_kernels import MAPPINGS, X_COLUMNS, X_ROWS
from ai_journey.memory_access_pattern import WarpAccess
from ai_journey.memory_access_tiling import MatmulShape

RAGGED = MatmulShape(m=37, n=45, k=29)
ALIGNED = MatmulShape(m=32, n=64, k=16)


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize(
    "shape",
    [RAGGED, ALIGNED, MatmulShape(1, 1, 1), MatmulShape(70, 3, 2)],
    ids=lambda shape: f"{shape.m}x{shape.n}x{shape.k}",
)
def test_both_labs_reproduce_every_request_and_warp_count(shape, mapping):
    launch = MatmulLaunch(shape, (32, 8), mapping)
    check = cross_check_launch(launch)
    full, partial, idle = check.warp_classes

    assert check.passed
    assert check.agreeing == check.requests == (full + partial) * (2 * shape.k + 1)
    assert check.mismatched_warps == 0
    assert full + partial + idle == launch.config.blocks * launch.warps_per_block
    assert check.to_mapping()["passed"] is True


def test_the_simt_coverage_model_counts_the_ragged_warps():
    assert simt_warp_classes(MatmulLaunch(RAGGED, (32, 8), X_COLUMNS)) == (37, 37, 6)
    assert simt_warp_classes(MatmulLaunch(RAGGED, (32, 8), X_ROWS)) == (45, 45, 6)
    assert simt_warp_classes(MatmulLaunch(ALIGNED, (32, 8), X_ROWS)) == (64, 0, 0)


def test_generators_walk_a_row_or_a_column_of_each_matrix():
    columns = MatmulLaunch(RAGGED, (32, 8), X_COLUMNS)
    rows = MatmulLaunch(RAGGED, (32, 8), X_ROWS)
    edge = WarpRef(Index3(1, 4), 4)

    assert memory_access_requests(columns, edge) == warp_requests(columns, edge)
    assert memory_access_requests(rows, edge) == warp_requests(rows, edge)
    assert memory_access_requests(columns, WarpRef(Index3(0, 4), 5)) is None


def test_a_model_that_drifts_from_the_layouts_fails_the_cross_check(monkeypatch):
    launch = MatmulLaunch(RAGGED, (32, 8), X_COLUMNS)
    honest = cross_check_launch(launch)

    def shifted(current, ref):
        requests = warp_requests(current, ref)
        if requests is None:
            return None
        moved = WarpAccess(
            4,
            tuple(
                None if address is None else address + 4
                for address in requests.b_loads[0].lane_addresses
            ),
        )
        return replace(requests, b_loads=(moved, *requests.b_loads[1:]))

    monkeypatch.setattr(matmul_model_crosscheck, "warp_requests", shifted)
    drifted = cross_check_launch(launch)

    # One B load of each of the 74 active warps moves by one element.
    assert drifted.requests == honest.requests
    assert drifted.agreeing == honest.agreeing - 74
    assert not drifted.passed

    monkeypatch.setattr(matmul_model_crosscheck, "warp_requests", lambda *_: None)
    idle = cross_check_launch(launch)
    assert (idle.requests, idle.mismatched_warps) == (0, 74)
    assert not idle.passed
