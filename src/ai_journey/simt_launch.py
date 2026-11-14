"""CUDA launch geometry checked against declared compute-capability limits.

Grids and blocks follow the CUDA execution configuration: a block's threads are
numbered x-fastest and grouped into warps of consecutive thread IDs, so a block
whose thread count is not a warp multiple leaves idle lanes in its last warp.
Nothing is launched; the checks only compare declared shapes with declared
limits.
"""

from __future__ import annotations

from dataclasses import dataclass

from .simt_device import SimtArchitecture, SimtError, require_int

_MAX_DIMENSION = 2**31 - 1


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


@dataclass(frozen=True)
class Dim3:
    """A positive CUDA ``dim3`` extent; unspecified axes default to one."""

    x: int
    y: int = 1
    z: int = 1

    def __post_init__(self) -> None:
        for axis in ("x", "y", "z"):
            require_int(
                getattr(self, axis), f"dim3.{axis}", minimum=1, maximum=_MAX_DIMENSION
            )

    @property
    def volume(self) -> int:
        return self.x * self.y * self.z

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.z]


@dataclass(frozen=True)
class LaunchConfiguration:
    """A grid of identical thread blocks."""

    grid: Dim3
    block: Dim3

    def __post_init__(self) -> None:
        if not isinstance(self.grid, Dim3) or not isinstance(self.block, Dim3):
            raise TypeError("grid and block must be Dim3")

    @property
    def threads_per_block(self) -> int:
        return self.block.volume

    @property
    def total_blocks(self) -> int:
        return self.grid.volume

    @property
    def total_threads(self) -> int:
        return self.threads_per_block * self.total_blocks


@dataclass(frozen=True)
class LaunchGeometry:
    """A launch admitted by one architecture, with its warp decomposition."""

    launch: LaunchConfiguration
    warp_size: int
    warps_per_block: int
    idle_lanes_per_block: int

    @property
    def threads_per_block(self) -> int:
        return self.launch.threads_per_block

    @property
    def total_blocks(self) -> int:
        return self.launch.total_blocks

    @property
    def total_threads(self) -> int:
        return self.launch.total_threads

    @property
    def total_warps(self) -> int:
        return self.warps_per_block * self.total_blocks

    @property
    def total_idle_lanes(self) -> int:
        return self.idle_lanes_per_block * self.total_blocks

    @property
    def lane_utilization(self) -> float:
        """Return active lanes divided by lanes in the block's issued warps."""

        return self.threads_per_block / (self.warps_per_block * self.warp_size)

    def to_mapping(self) -> dict[str, object]:
        return {
            "grid": self.launch.grid.to_list(),
            "block": self.launch.block.to_list(),
            "threads_per_block": self.threads_per_block,
            "warps_per_block": self.warps_per_block,
            "idle_lanes_per_block": self.idle_lanes_per_block,
            "total_blocks": self.total_blocks,
            "total_threads": self.total_threads,
            "total_warps": self.total_warps,
            "total_idle_lanes": self.total_idle_lanes,
            "lane_utilization": self.lane_utilization,
        }


def check_launch(
    architecture: SimtArchitecture, launch: LaunchConfiguration
) -> LaunchGeometry:
    """Reject launches outside the declared block and grid dimension limits."""

    if not isinstance(architecture, SimtArchitecture):
        raise TypeError("architecture must be SimtArchitecture")
    if not isinstance(launch, LaunchConfiguration):
        raise TypeError("launch must be LaunchConfiguration")
    capability = architecture.compute_capability
    for label, extent, limits in (
        (
            "block",
            launch.block,
            (
                architecture.max_block_dim_x,
                architecture.max_block_dim_y,
                architecture.max_block_dim_z,
            ),
        ),
        (
            "grid",
            launch.grid,
            (
                architecture.max_grid_dim_x,
                architecture.max_grid_dim_y,
                architecture.max_grid_dim_z,
            ),
        ),
    ):
        for axis, value, limit in zip("xyz", extent.to_list(), limits):
            if value > limit:
                raise SimtError(
                    f"{label} {axis} dimension {value} exceeds the {limit} limit "
                    f"for compute capability {capability}"
                )
    if launch.threads_per_block > architecture.max_threads_per_block:
        raise SimtError(
            f"block has {launch.threads_per_block} threads; compute capability "
            f"{capability} allows {architecture.max_threads_per_block}"
        )
    warps = _ceil_div(launch.threads_per_block, architecture.warp_size)
    return LaunchGeometry(
        launch=launch,
        warp_size=architecture.warp_size,
        warps_per_block=warps,
        idle_lanes_per_block=warps * architecture.warp_size - launch.threads_per_block,
    )


def launch_for_elements(
    architecture: SimtArchitecture, *, elements: int, threads_per_block: int
) -> LaunchGeometry:
    """Size a 1D grid with ceil division so every element gets one thread."""

    count = require_int(elements, "elements", minimum=1, maximum=2**62)
    block = require_int(
        threads_per_block, "threads_per_block", minimum=1, maximum=_MAX_DIMENSION
    )
    blocks = _ceil_div(count, block)
    if blocks > _MAX_DIMENSION:
        raise SimtError("elements need more blocks than one grid dimension holds")
    return check_launch(architecture, LaunchConfiguration(Dim3(blocks), Dim3(block)))


def launch_for_image(
    architecture: SimtArchitecture,
    *,
    width: int,
    height: int,
    block_x: int,
    block_y: int,
) -> LaunchGeometry:
    """Size a 2D grid with ceil division over image columns (x) and rows (y)."""

    columns = require_int(width, "width", minimum=1, maximum=_MAX_DIMENSION)
    rows = require_int(height, "height", minimum=1, maximum=_MAX_DIMENSION)
    extent_x = require_int(block_x, "block_x", minimum=1, maximum=_MAX_DIMENSION)
    extent_y = require_int(block_y, "block_y", minimum=1, maximum=_MAX_DIMENSION)
    return check_launch(
        architecture,
        LaunchConfiguration(
            Dim3(_ceil_div(columns, extent_x), _ceil_div(rows, extent_y)),
            Dim3(extent_x, extent_y),
        ),
    )
