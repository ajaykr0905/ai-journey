"""Strict CUDA launch configurations under one stated compute capability.

A launch is a grid of identical thread blocks. ``LaunchConfig`` admits a grid
and a block only inside the compute capability 8.0 limits named below. The
values are copied from the NVIDIA CUDA C++ Programming Guide; nothing here
detects or queries a GPU. A rejected launch raises ``LaunchLimitError`` with
the name of the limit it breaks, so an oversized block never reads like an
oversized grid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

COMPUTE_CAPABILITY = "8.0"
WARP_SIZE = 32
MAX_THREADS_PER_BLOCK = 1024
MAX_BLOCK_DIM_X = 1024
MAX_BLOCK_DIM_Y = 1024
MAX_BLOCK_DIM_Z = 64
MAX_GRID_DIM_X = 2**31 - 1
MAX_GRID_DIM_Y = 65_535
MAX_GRID_DIM_Z = 65_535
LIMITS_SOURCE = (
    "NVIDIA CUDA C++ Programming Guide v12.6 (archive 12.6.3), Table 21 "
    "Technical Specifications per Compute Capability, compute capability 8.0"
)

# CUDA declares the components of dim3 and uint3 as unsigned int.
UINT_MAX = 2**32 - 1

_BLOCK_LIMITS = (
    ("max_block_dim_x", MAX_BLOCK_DIM_X),
    ("max_block_dim_y", MAX_BLOCK_DIM_Y),
    ("max_block_dim_z", MAX_BLOCK_DIM_Z),
)
_GRID_LIMITS = (
    ("max_grid_dim_x", MAX_GRID_DIM_X),
    ("max_grid_dim_y", MAX_GRID_DIM_Y),
    ("max_grid_dim_z", MAX_GRID_DIM_Z),
)

LAUNCH_LIMITS: Mapping[str, int] = MappingProxyType(
    {
        "warp_size": WARP_SIZE,
        "max_threads_per_block": MAX_THREADS_PER_BLOCK,
        **dict(_BLOCK_LIMITS),
        **dict(_GRID_LIMITS),
    }
)


class KernelLaunchError(ValueError):
    """Raised when a kernel-launch lab input, run or report breaks its contract."""


class LaunchLimitError(KernelLaunchError):
    """Raised when one launch value exceeds a named compute-capability limit.

    ``limit`` holds the name of the broken limit, such as ``max_block_dim_z``,
    so a caller can tell which rule failed without parsing the message.
    """

    def __init__(self, *, limit: str, subject: str, value: int) -> None:
        maximum = LAUNCH_LIMITS[limit]
        super().__init__(
            f"{subject} = {value} breaks {limit} = {maximum} for compute "
            f"capability {COMPUTE_CAPABILITY}"
        )
        self.limit = limit
        self.value = value
        self.maximum = maximum


def require_int(value: object, name: str, *, minimum: int, maximum: int) -> int:
    """Return an exact integer inside explicit bounds; booleans are rejected."""

    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise KernelLaunchError(f"{name} must be between {minimum} and {maximum}")
    return value


def ceil_div(numerator: int, denominator: int) -> int:
    """Return how many ``denominator``-sized chunks cover ``numerator`` items.

    This is the lecture's ``cdiv``, ``(numerator + denominator - 1) //
    denominator``: the smallest block count that leaves no item without a
    thread.
    """

    require_int(numerator, "numerator", minimum=0, maximum=2**64)
    require_int(denominator, "denominator", minimum=1, maximum=2**64)
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class Dim3:
    """A CUDA ``dim3`` extent: positive components, and unset axes are one."""

    x: int
    y: int = 1
    z: int = 1

    def __post_init__(self) -> None:
        for axis in "xyz":
            require_int(
                getattr(self, axis), f"dim3.{axis}", minimum=1, maximum=UINT_MAX
            )

    @property
    def volume(self) -> int:
        return self.x * self.y * self.z

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, value: object, label: str) -> Dim3:
        """Parse a JSON ``[x, y, z]`` array without defaults or coercion."""

        if type(value) is not list or len(value) != 3:
            raise KernelLaunchError(f"{label} must be a JSON array [x, y, z]")
        return cls(*value)


@dataclass(frozen=True)
class Index3:
    """A CUDA ``uint3`` index such as ``blockIdx`` or ``threadIdx``."""

    x: int
    y: int = 0
    z: int = 0

    def __post_init__(self) -> None:
        for axis in "xyz":
            require_int(
                getattr(self, axis), f"index.{axis}", minimum=0, maximum=UINT_MAX - 1
            )

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.z]


@dataclass(frozen=True)
class LaunchConfig:
    """A grid of identical thread blocks that the named limits admit.

    Checks run in a fixed order: each block axis, then the threads in one
    block, then each grid axis. The first broken limit is the one reported.
    """

    grid: Dim3
    block: Dim3

    def __post_init__(self) -> None:
        if type(self.grid) is not Dim3 or type(self.block) is not Dim3:
            raise TypeError("grid and block must be Dim3")
        for axis, value, (limit, maximum) in zip(
            "xyz", self.block.to_list(), _BLOCK_LIMITS
        ):
            if value > maximum:
                raise LaunchLimitError(
                    limit=limit, subject=f"block.{axis}", value=value
                )
        if self.block.volume > MAX_THREADS_PER_BLOCK:
            raise LaunchLimitError(
                limit="max_threads_per_block",
                subject="block.x * block.y * block.z",
                value=self.block.volume,
            )
        for axis, value, (limit, maximum) in zip(
            "xyz", self.grid.to_list(), _GRID_LIMITS
        ):
            if value > maximum:
                raise LaunchLimitError(limit=limit, subject=f"grid.{axis}", value=value)

    @property
    def threads_per_block(self) -> int:
        return self.block.volume

    @property
    def blocks(self) -> int:
        return self.grid.volume

    @property
    def launched_threads(self) -> int:
        return self.threads_per_block * self.blocks

    @property
    def warps_per_block(self) -> int:
        return ceil_div(self.threads_per_block, WARP_SIZE)

    @property
    def idle_lanes_per_block(self) -> int:
        """Return lanes of the last warp that hold no thread of the block."""

        return self.warps_per_block * WARP_SIZE - self.threads_per_block

    def threads_inside(self, extent: Dim3) -> int:
        """Count threads whose global (x, y, z) index lies inside ``extent``.

        These are the threads that pass a bounds guard on every axis. The
        other launched threads are idle: they do no useful work.
        """

        if type(extent) is not Dim3:
            raise TypeError("extent must be Dim3")
        count = 1
        for blocks, threads, size in zip(
            self.grid.to_list(), self.block.to_list(), extent.to_list()
        ):
            count *= min(blocks * threads, size)
        return count

    def thread_id(self, block: Index3, thread: Index3) -> int:
        """Return the linear ID ``block_id * threads_per_block + local_id``.

        CUDA numbers the threads of a block x fastest, then y, then z. This
        lab numbers the blocks of a grid the same way, so ascending linear IDs
        are the simulator's canonical order.
        """

        if type(block) is not Index3 or type(thread) is not Index3:
            raise TypeError("block and thread must be Index3")
        for label, index, extent in (
            ("block", block, self.grid),
            ("thread", thread, self.block),
        ):
            for axis, value, size in zip("xyz", index.to_list(), extent.to_list()):
                if value >= size:
                    raise KernelLaunchError(
                        f"{label} index {axis} = {value} is outside its extent {size}"
                    )
        block_id = block.x + self.grid.x * (block.y + self.grid.y * block.z)
        thread_index = thread.x + self.block.x * (thread.y + self.block.y * thread.z)
        return block_id * self.threads_per_block + thread_index

    def coordinates(self, thread_id: int) -> tuple[Index3, Index3]:
        """Return the ``(blockIdx, threadIdx)`` pair of a linear thread ID."""

        require_int(
            thread_id, "thread_id", minimum=0, maximum=self.launched_threads - 1
        )
        block_id, local = divmod(thread_id, self.threads_per_block)
        return _unflatten(block_id, self.grid), _unflatten(local, self.block)

    def to_mapping(self) -> dict[str, object]:
        return {"grid": self.grid.to_list(), "block": self.block.to_list()}

    def geometry(self) -> dict[str, object]:
        """Return the launch shape and its derived thread and warp counts."""

        return {
            **self.to_mapping(),
            "threads_per_block": self.threads_per_block,
            "blocks": self.blocks,
            "launched_threads": self.launched_threads,
            "warps_per_block": self.warps_per_block,
            "idle_lanes_per_block": self.idle_lanes_per_block,
        }

    @classmethod
    def from_mapping(cls, mapping: object) -> LaunchConfig:
        """Parse ``{"grid": [x, y, z], "block": [x, y, z]}`` with no extra field."""

        if not isinstance(mapping, Mapping) or set(mapping) != {"grid", "block"}:
            raise KernelLaunchError("launch config must hold exactly grid and block")
        return cls(
            Dim3.from_list(mapping["grid"], "grid"),
            Dim3.from_list(mapping["block"], "block"),
        )


def _unflatten(linear: int, extent: Dim3) -> Index3:
    upper, x = divmod(linear, extent.x)
    z, y = divmod(upper, extent.y)
    return Index3(x, y, z)


def grid_for(extent: Dim3, block: Dim3) -> LaunchConfig:
    """Size the smallest grid of ``block``-shaped blocks that covers ``extent``.

    Each grid axis is ``ceil_div(extent, block)``. When an extent is not a
    multiple of the block on some axis, the last block on that axis reaches
    past the extent, so the kernel needs a bounds guard.
    """

    if type(extent) is not Dim3 or type(block) is not Dim3:
        raise TypeError("extent and block must be Dim3")
    axes = [
        ceil_div(size, threads)
        for size, threads in zip(extent.to_list(), block.to_list())
    ]
    return LaunchConfig(Dim3(*axes), block)


def launch_1d(n: int, threads_per_block: int) -> LaunchConfig:
    """Size a 1D launch with one thread per element: ``cdiv(n, threads)``."""

    require_int(n, "n", minimum=1, maximum=UINT_MAX)
    require_int(threads_per_block, "threads_per_block", minimum=1, maximum=UINT_MAX)
    return grid_for(Dim3(n), Dim3(threads_per_block))


def launch_2d(width: int, height: int, block_x: int, block_y: int) -> LaunchConfig:
    """Size a 2D launch: x runs over image columns and y over image rows."""

    for name, value in (
        ("width", width),
        ("height", height),
        ("block_x", block_x),
        ("block_y", block_y),
    ):
        require_int(value, name, minimum=1, maximum=UINT_MAX)
    return grid_for(Dim3(width, height), Dim3(block_x, block_y))
