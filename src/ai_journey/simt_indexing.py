"""Exact global-index coverage for 1D and 2D row-major CUDA launches.

A 1D thread maps to ``blockIdx.x * blockDim.x + threadIdx.x``. A 2D thread maps
to ``column = blockIdx.x * blockDim.x + threadIdx.x`` and ``row = blockIdx.y *
blockDim.y + threadIdx.y`` with row-major linear index ``row * width + column``.
Counts are closed-form or enumerate one representative block per edge class,
so they stay exact for full-size grids without visiting every thread.
"""

from __future__ import annotations

from dataclasses import dataclass

from .simt_device import SimtError, require_int
from .simt_launch import LaunchConfiguration, LaunchGeometry

_MAX_EXTENT = 2**31 - 1


def global_index_1d(*, block_index: int, block_dim: int, thread_index: int) -> int:
    """Return ``blockIdx.x * blockDim.x + threadIdx.x`` for one thread."""

    dim = require_int(block_dim, "block_dim", minimum=1, maximum=_MAX_EXTENT)
    block = require_int(block_index, "block_index", minimum=0, maximum=_MAX_EXTENT)
    thread = require_int(thread_index, "thread_index", minimum=0, maximum=dim - 1)
    return block * dim + thread


def global_index_2d(
    *,
    block_index: tuple[int, int],
    block_dim: tuple[int, int],
    thread_index: tuple[int, int],
    width: int,
) -> tuple[int, int, int]:
    """Return ``(row, column, row * width + column)`` for one 2D thread."""

    require_int(width, "width", minimum=1, maximum=_MAX_EXTENT)
    for label, value in (
        ("block_index", block_index),
        ("block_dim", block_dim),
        ("thread_index", thread_index),
    ):
        if type(value) is not tuple or len(value) != 2:
            raise SimtError(f"{label} must be an (x, y) tuple")
    column = global_index_1d(
        block_index=block_index[0], block_dim=block_dim[0], thread_index=thread_index[0]
    )
    row = global_index_1d(
        block_index=block_index[1], block_dim=block_dim[1], thread_index=thread_index[1]
    )
    return row, column, row * width + column


@dataclass(frozen=True)
class IndexCoverage:
    """Element coverage and guard behavior of one launch over one data shape.

    ``masked_threads`` fail the bounds guard and are disabled lanes when the
    guard is present. Without a guard, each of them writes either beyond the
    buffer (``unguarded_out_of_bounds_writes``) or, in 2D, onto another
    element's in-bounds slot (``unguarded_aliased_writes``). Warp counts
    classify warps by their lanes under the guard; idle lanes of partial warps
    are not threads and are excluded.
    """

    layout: str
    width: int
    height: int
    launched_threads: int
    covered_elements: int
    uncovered_elements: int
    masked_threads: int
    unguarded_out_of_bounds_writes: int
    unguarded_aliased_writes: int
    fully_active_warps: int
    partially_masked_warps: int
    fully_masked_warps: int

    @property
    def elements(self) -> int:
        return self.width * self.height

    @property
    def under_covered(self) -> bool:
        return self.uncovered_elements > 0

    @property
    def needs_bounds_guard(self) -> bool:
        return self.masked_threads > 0

    def to_mapping(self) -> dict[str, object]:
        return {
            "layout": self.layout,
            "width": self.width,
            "height": self.height,
            "elements": self.elements,
            "launched_threads": self.launched_threads,
            "covered_elements": self.covered_elements,
            "uncovered_elements": self.uncovered_elements,
            "masked_threads": self.masked_threads,
            "unguarded_out_of_bounds_writes": self.unguarded_out_of_bounds_writes,
            "unguarded_aliased_writes": self.unguarded_aliased_writes,
            "fully_active_warps": self.fully_active_warps,
            "partially_masked_warps": self.partially_masked_warps,
            "fully_masked_warps": self.fully_masked_warps,
            "under_covered": self.under_covered,
            "needs_bounds_guard": self.needs_bounds_guard,
        }


def _axis_classes(block_extent: int, blocks: int, limit: int) -> list[tuple[int, int]]:
    """Split one axis into (block count, in-bounds threads per block) classes."""

    inside = min(blocks, limit // block_extent)
    classes = [(inside, block_extent)] if inside else []
    remaining = blocks - inside
    if remaining and inside * block_extent < limit:
        classes.append((1, limit - inside * block_extent))
        remaining -= 1
    if remaining:
        classes.append((remaining, 0))
    return classes


def _out_of_bounds(columns: int, rows: int, width: int, height: int) -> int:
    """Count threads whose unguarded row-major index is at least width * height."""

    below = max(0, rows - height) * columns
    # A row r < height spills past the buffer only from column (height - r) *
    # width; with k = height - r that leaves max(0, columns - k * width) threads.
    lowest = height - min(rows, height) + 1
    highest = min(height, (columns - 1) // width)
    count = max(0, highest - lowest + 1)
    spill = count * columns - width * (lowest + highest) * count // 2
    return below + spill


def _warp_classes(
    launch: LaunchConfiguration, width: int, height: int, warp_size: int
) -> tuple[int, int, int]:
    block_x, block_y = launch.block.x, launch.block.y
    threads = block_x * block_y
    active = partial = masked = 0
    for x_blocks, valid_x in _axis_classes(block_x, launch.grid.x, width):
        for y_blocks, valid_y in _axis_classes(block_y, launch.grid.y, height):
            multiplicity = x_blocks * y_blocks
            for first in range(0, threads, warp_size):
                lanes = range(first, min(first + warp_size, threads))
                passing = sum(
                    1
                    for thread in lanes
                    if thread % block_x < valid_x and thread // block_x < valid_y
                )
                if passing == len(lanes):
                    active += multiplicity
                elif passing:
                    partial += multiplicity
                else:
                    masked += multiplicity
    return active, partial, masked


def _coverage(
    launch: LaunchConfiguration,
    *,
    layout: str,
    width: int,
    height: int,
    warp_size: int,
) -> IndexCoverage:
    columns = launch.grid.x * launch.block.x
    rows = launch.grid.y * launch.block.y
    covered = min(columns, width) * min(rows, height)
    launched = columns * rows
    masked = launched - covered
    out_of_bounds = _out_of_bounds(columns, rows, width, height)
    active, partial, fully_masked = _warp_classes(launch, width, height, warp_size)
    return IndexCoverage(
        layout=layout,
        width=width,
        height=height,
        launched_threads=launched,
        covered_elements=covered,
        uncovered_elements=width * height - covered,
        masked_threads=masked,
        unguarded_out_of_bounds_writes=out_of_bounds,
        unguarded_aliased_writes=masked - out_of_bounds,
        fully_active_warps=active,
        partially_masked_warps=partial,
        fully_masked_warps=fully_masked,
    )


def _require_geometry(geometry: object) -> LaunchGeometry:
    if not isinstance(geometry, LaunchGeometry):
        raise TypeError("geometry must be a LaunchGeometry admitted by check_launch")
    return geometry


def coverage_1d(geometry: LaunchGeometry, *, elements: int) -> IndexCoverage:
    """Analyze an admitted 1D launch over ``elements`` contiguous items."""

    launch = _require_geometry(geometry).launch
    count = require_int(elements, "elements", minimum=1, maximum=2**62)
    if (launch.grid.y, launch.grid.z, launch.block.y, launch.block.z) != (1, 1, 1, 1):
        raise SimtError("1D coverage requires a one-dimensional grid and block")
    return _coverage(
        launch, layout="1d", width=count, height=1, warp_size=geometry.warp_size
    )


def coverage_2d(geometry: LaunchGeometry, *, width: int, height: int) -> IndexCoverage:
    """Analyze an admitted 2D launch over a row-major ``height x width`` array."""

    launch = _require_geometry(geometry).launch
    columns = require_int(width, "width", minimum=1, maximum=_MAX_EXTENT)
    rows = require_int(height, "height", minimum=1, maximum=_MAX_EXTENT)
    if launch.grid.z != 1 or launch.block.z != 1:
        raise SimtError("2D coverage requires z extents of one")
    return _coverage(
        launch,
        layout="2d_row_major",
        width=columns,
        height=rows,
        warp_size=geometry.warp_size,
    )
