"""Metal launch geometry: whole threadgroups by ceiling division, checked first.

``torch.mps.compile_shader`` launches a kernel over ``threads`` grid threads
per axis in threadgroups of ``group_size`` threads, the Metal counterparts of
a CUDA grid and block. A CUDA launch for ``n`` elements uses ``ceil(n /
block)`` blocks; this module derives the same group counts and passes
``groups * group_size`` threads per axis, so the last threadgroup on an axis
can hold padding threads that the kernel's bounds guard turns away, as in
CUDA. Axis 0 is ``x``, the fastest-varying axis: the element index in 1D and
the image column in 2D.

The API exposes limits only through the compiled pipeline state: the most
threads one threadgroup may hold and the SIMD-group width, Metal's
counterpart of a CUDA warp. Every launch is checked against the pipeline that
will run it before it runs; nothing is launched here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


class KernelMetalError(ValueError):
    """Raised when a Metal kernel control, launch, check or report is invalid."""


MAX_AXES = 2
# thread_position_in_grid is a 32-bit uint on every axis.
MAX_GRID_THREADS_PER_AXIS = 2**32 - 1
# Metal's documented ceiling for one threadgroup on Apple GPUs; a compiled
# pipeline can report a lower limit, which ``CheckedLaunch`` enforces.
MAX_THREADGROUP_THREADS = 1024
MAX_THREADGROUP_MEMORY_BYTES = 1 << 20


def require_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    """Return a true integer inside ``[minimum, maximum]``.

    Every count passes through here, so JSON ``true`` or ``2.0`` can never
    stand in for one.
    """

    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise KernelMetalError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def ceil_div(numerator: int, denominator: int) -> int:
    """Return ``ceil(numerator / denominator)`` for positive integers, exactly."""

    return -(-numerator // denominator)


@dataclass(frozen=True)
class PipelineLimits:
    """Threadgroup limits that one compiled Metal pipeline reports about itself.

    These are the only device limits ``torch.mps.compile_shader`` exposes, so
    a launch is checked against the pipeline that runs it, not a table.
    """

    max_threads_per_threadgroup: int
    thread_execution_width: int
    static_threadgroup_memory_bytes: int

    def __post_init__(self) -> None:
        require_integer(
            self.max_threads_per_threadgroup,
            "max_threads_per_threadgroup",
            1,
            MAX_THREADGROUP_THREADS,
        )
        require_integer(
            self.thread_execution_width,
            "thread_execution_width",
            1,
            self.max_threads_per_threadgroup,
        )
        require_integer(
            self.static_threadgroup_memory_bytes,
            "static_threadgroup_memory_bytes",
            0,
            MAX_THREADGROUP_MEMORY_BYTES,
        )

    def to_mapping(self) -> dict[str, int]:
        """Return the limits as a report records them."""

        return {
            "max_threads_per_threadgroup": self.max_threads_per_threadgroup,
            "thread_execution_width": self.thread_execution_width,
            "static_threadgroup_memory_bytes": self.static_threadgroup_memory_bytes,
        }


@dataclass(frozen=True)
class LaunchGeometry:
    """Data extent and threadgroup size per axis; the grid is whole groups."""

    extent: tuple[int, ...]
    group_size: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.extent) is not tuple
            or type(self.group_size) is not tuple
            or not 1 <= len(self.extent) <= MAX_AXES
            or len(self.group_size) != len(self.extent)
        ):
            raise KernelMetalError(
                "extent and group_size must be tuples with the same 1 or 2 axes"
            )
        for axis, (size, group) in enumerate(zip(self.extent, self.group_size)):
            require_integer(size, f"extent[{axis}]", 1, MAX_GRID_THREADS_PER_AXIS)
            require_integer(group, f"group_size[{axis}]", 1, MAX_THREADGROUP_THREADS)
            if ceil_div(size, group) * group > MAX_GRID_THREADS_PER_AXIS:
                raise KernelMetalError(
                    f"axis {axis} needs more than {MAX_GRID_THREADS_PER_AXIS} "
                    "grid threads"
                )
        if prod(self.group_size) > MAX_THREADGROUP_THREADS:
            raise KernelMetalError(
                f"a threadgroup of {prod(self.group_size)} threads exceeds "
                f"Metal's {MAX_THREADGROUP_THREADS}-thread ceiling"
            )

    @property
    def groups(self) -> tuple[int, ...]:
        """Threadgroups per axis, ``ceil(extent / group_size)``: CUDA's grid size."""

        return tuple(
            ceil_div(size, group) for size, group in zip(self.extent, self.group_size)
        )

    @property
    def threads(self) -> tuple[int, ...]:
        """Grid threads per axis, rounded up to whole threadgroups."""

        return tuple(
            count * group for count, group in zip(self.groups, self.group_size)
        )

    @property
    def threads_per_group(self) -> int:
        """Threads in one threadgroup: CUDA's threads per block."""

        return prod(self.group_size)

    @property
    def launched_threads(self) -> int:
        """Every thread of the grid, including the padding threads."""

        return prod(self.threads)

    @property
    def active_threads(self) -> int:
        """Threads that pass the bounds guard: one per element or pixel."""

        return prod(self.extent)

    @property
    def padding_threads(self) -> int:
        """Threads past the data on some axis, which the bounds guard stops."""

        return self.launched_threads - self.active_threads

    def launch_arguments(self) -> dict[str, object]:
        """Return the ``threads`` and ``group_size`` keywords compile_shader takes.

        One axis passes plain integers; two axes pass ``[x, y]`` lists.
        """

        if len(self.extent) == 1:
            return {"threads": self.threads[0], "group_size": self.group_size[0]}
        return {"threads": list(self.threads), "group_size": list(self.group_size)}

    def to_mapping(self) -> dict[str, object]:
        """Return the geometry a report records, so a verifier can rederive it."""

        return {
            "extent": list(self.extent),
            "group_size": list(self.group_size),
            "groups": list(self.groups),
            "threads": list(self.threads),
            "threads_per_group": self.threads_per_group,
            "launched_threads": self.launched_threads,
            "active_threads": self.active_threads,
            "padding_threads": self.padding_threads,
        }


@dataclass(frozen=True)
class CheckedLaunch:
    """A launch admitted by one pipeline, with its SIMD-group decomposition.

    A threadgroup runs as SIMD groups of ``thread_execution_width`` threads,
    so a group size that is not a multiple of the width leaves idle lanes in
    its last SIMD group, as a CUDA block does in its last warp.
    """

    geometry: LaunchGeometry
    limits: PipelineLimits

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, LaunchGeometry):
            raise TypeError("geometry must be a LaunchGeometry")
        if not isinstance(self.limits, PipelineLimits):
            raise TypeError("limits must be PipelineLimits")
        threads = self.geometry.threads_per_group
        limit = self.limits.max_threads_per_threadgroup
        if threads > limit:
            raise KernelMetalError(
                f"a threadgroup of {threads} threads exceeds this pipeline's "
                f"limit of {limit}"
            )

    @property
    def simd_groups_per_threadgroup(self) -> int:
        """SIMD groups one threadgroup occupies, rounded up."""

        return ceil_div(
            self.geometry.threads_per_group, self.limits.thread_execution_width
        )

    @property
    def idle_lanes_per_threadgroup(self) -> int:
        """Lanes of the last SIMD group that hold no thread of the group."""

        width = self.limits.thread_execution_width
        return (
            self.simd_groups_per_threadgroup * width - self.geometry.threads_per_group
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the geometry and its SIMD decomposition for a report."""

        return {
            **self.geometry.to_mapping(),
            "simd_groups_per_threadgroup": self.simd_groups_per_threadgroup,
            "idle_lanes_per_threadgroup": self.idle_lanes_per_threadgroup,
        }


def vector_launch(length: int, group_size: int) -> LaunchGeometry:
    """Size a 1D launch with one thread per element, like ``ceil(n / block)``."""

    return LaunchGeometry(extent=(length,), group_size=(group_size,))


def image_launch(
    height: int, width: int, group_size: tuple[int, int]
) -> LaunchGeometry:
    """Size a 2D launch with one thread per pixel: x over columns, y over rows.

    ``group_size`` is ``(x, y)``, the order Metal and CUDA both use.
    """

    if type(group_size) is not tuple or len(group_size) != 2:
        raise KernelMetalError("an image group_size must be an (x, y) tuple")
    return LaunchGeometry(extent=(width, height), group_size=group_size)
