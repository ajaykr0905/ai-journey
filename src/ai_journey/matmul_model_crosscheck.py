"""Cross-checks of the warp request model against two earlier labs.

The access model derives each lane address from the kernel's index
expressions. ``memory_access_requests`` rebuilds the same requests from the
memory_access lab's matrix layouts and generators instead. A warp that walks
a row of a matrix is a ``row_access``, a warp that walks a column is a
``column_access``, and a warp whose lanes share one element is a
``broadcast_access`` masked to the lanes that the guard keeps. That lab's
``analyze_coalescing`` then counts the sectors and lines of each request.

``cross_check_launch`` requires both labs to give the same lane addresses and
the same counts for every request of a launch. It also asks the SIMT lab's
coverage model how many warps the guard leaves full, partly masked and fully
masked, and compares those counts with the access model's.
"""

from __future__ import annotations

from dataclasses import dataclass

from .matmul_model_access import (
    MatmulLaunch,
    WarpRef,
    WarpRequests,
    lane_elements,
    line_indices,
    sector_indices,
    warp_refs,
    warp_requests,
)
from .matmul_model_kernels import X_COLUMNS, thread_extent
from .memory_access_coalescing import analyze_coalescing
from .memory_access_pattern import (
    WARP_SIZE,
    MatrixLayout,
    WarpAccess,
    broadcast_access,
    column_access,
    lane_mask,
    row_access,
)
from .simt_device import PINNED_ARCHITECTURES
from .simt_indexing import coverage_2d
from .simt_launch import launch_for_image

SIMT_COMPUTE_CAPABILITY = "8.0"


def memory_access_requests(launch: MatmulLaunch, ref: WarpRef) -> WarpRequests | None:
    """Rebuild one warp's requests with the memory_access lab's generators."""

    shape = launch.shape
    m, n, k = shape.m, shape.n, shape.k
    lanes = lane_elements(launch, ref)
    active = [lane for lane, (row, col) in enumerate(lanes) if row < m and col < n]
    if not active:
        return None
    mask = lane_mask(active)
    row, col = lanes[active[0]]
    a_layout = MatrixLayout(rows=m, columns=k, leading_dimension=k)
    b_layout = MatrixLayout(rows=k, columns=n, leading_dimension=n)
    c_layout = MatrixLayout(rows=m, columns=n, leading_dimension=n)
    if launch.mapping == X_COLUMNS:
        a_loads = tuple(
            broadcast_access(address_bytes=a_layout.address(row, i)).masked(mask)
            for i in range(k)
        )
        b_loads = tuple(row_access(b_layout, row=i, start_column=col) for i in range(k))
        c_store = row_access(c_layout, row=row, start_column=col)
    else:
        a_loads = tuple(
            column_access(a_layout, column=i, start_row=row) for i in range(k)
        )
        b_loads = tuple(
            broadcast_access(address_bytes=b_layout.address(i, col)).masked(mask)
            for i in range(k)
        )
        c_store = column_access(c_layout, column=col, start_row=row)
    return WarpRequests(ref, len(active), a_loads, b_loads, c_store)


def _requests(warp: WarpRequests) -> tuple[WarpAccess, ...]:
    return (*warp.a_loads, *warp.b_loads, warp.c_store)


def _agree(model: WarpAccess, other: WarpAccess) -> bool:
    counted = analyze_coalescing(other)
    return (
        model.lane_addresses == other.lane_addresses
        and len(sector_indices(model)) == counted.sector_count
        and len(line_indices(model)) == counted.line_count
    )


def simt_warp_classes(launch: MatmulLaunch) -> tuple[int, int, int]:
    """Return the SIMT lab's full, partly masked and fully masked warp counts."""

    extent = thread_extent(launch.shape, launch.mapping)
    block_x, block_y = launch.block
    geometry = launch_for_image(
        PINNED_ARCHITECTURES[SIMT_COMPUTE_CAPABILITY],
        width=extent.x,
        height=extent.y,
        block_x=block_x,
        block_y=block_y,
    )
    coverage = coverage_2d(geometry, width=extent.x, height=extent.y)
    return (
        coverage.fully_active_warps,
        coverage.partially_masked_warps,
        coverage.fully_masked_warps,
    )


@dataclass(frozen=True)
class CrossCheck:
    """How many requests and warp counts two other labs reproduce."""

    requests: int
    agreeing: int
    mismatched_warps: int
    warp_classes: tuple[int, int, int]
    simt_warp_classes: tuple[int, int, int]

    @property
    def passed(self) -> bool:
        return (
            self.agreeing == self.requests
            and self.mismatched_warps == 0
            and self.warp_classes == self.simt_warp_classes
        )

    def to_mapping(self) -> dict[str, object]:
        names = ("full", "partial", "idle")
        return {
            "memory_access_requests": self.requests,
            "memory_access_agreeing": self.agreeing,
            "mismatched_warps": self.mismatched_warps,
            "warp_classes": dict(zip(names, self.warp_classes)),
            "simt_warp_classes": dict(zip(names, self.simt_warp_classes)),
            "passed": self.passed,
        }


def cross_check_launch(launch: MatmulLaunch) -> CrossCheck:
    """Compare every request and the warp counts of a launch with both labs."""

    requests = agreeing = mismatched = full = partial = idle = 0
    for ref in warp_refs(launch):
        model = warp_requests(launch, ref)
        other = memory_access_requests(launch, ref)
        if (model is None) != (other is None):
            mismatched += 1
            continue
        if model is None:
            idle += 1
            continue
        if model.active_lanes == WARP_SIZE:
            full += 1
        else:
            partial += 1
        pairs = tuple(zip(_requests(model), _requests(other)))
        requests += len(pairs)
        agreeing += sum(_agree(left, right) for left, right in pairs)
    return CrossCheck(
        requests=requests,
        agreeing=agreeing,
        mismatched_warps=mismatched,
        warp_classes=(full, partial, idle),
        simt_warp_classes=simt_warp_classes(launch),
    )
