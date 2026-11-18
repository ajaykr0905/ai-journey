"""Admit or reject every launch shape of the sweep against the pipeline that runs it.

``torch.mps.compile_shader`` exposes threadgroup limits only through each
compiled pipeline: ``max_threads_per_threadgroup`` and
``thread_execution_width``, Metal's SIMD-group width. A shape is checked
against the pipeline of the kernel it will run, by two rules in order:

1. Its threads must fit ``max_threads_per_threadgroup``. Metal documents the
   limit, and PyTorch refuses a larger threadgroup with ``ValueError``.
2. Where the runtime requires whole SIMD groups, its threads must be a whole
   multiple of ``thread_execution_width``. A Metal pipeline requires that only
   when it was built with ``threadGroupSizeIsMultipleOfThreadExecutionWidth``;
   the runtime states whether its pipelines were.

A rejected shape keeps its reason and is never launched, so a sweep reports
what it refused instead of failing on the first shape a host cannot run. The
grid of an admitted shape is always whole threadgroups, by the ceiling
division of the Day 73 geometry, so the padding threads of the edge groups
meet the kernel's bounds guard.

Each admission also records how the threadgroup splits into SIMD groups under
the linear layout that the runtime's layout probe observes on the GPU: the
linear index is ``x + gx * y``, and each SIMD group holds ``width``
consecutive linear indices. The rows and columns that one SIMD group spans
tie a 2D shape to the execution width. With the x-over-columns mapping, they
are the output rows and columns that one SIMD group computes at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .kernel_metal_geometry import (
    MAX_THREADGROUP_THREADS,
    CheckedLaunch,
    KernelMetalError,
    LaunchGeometry,
    PipelineLimits,
    ceil_div,
    require_integer,
)

EXCEEDS_MAXIMUM = "exceeds_max_threads_per_threadgroup"
NOT_WHOLE_SIMD_GROUPS = "not_whole_simd_groups"
REJECTION_REASONS = (EXCEEDS_MAXIMUM, NOT_WHOLE_SIMD_GROUPS)
LINEAR_LAYOUT = "linear_x_fastest"
MAX_SHAPES = 16
MAX_MATRIX_SIDE = 4096


class MetalLaunchSweepError(KernelMetalError):
    """Raised when a launch-sweep control, launch, check or report is invalid."""


def group_shape(
    value: object, axes: int | None = None, label: str = "group_size"
) -> tuple[int, ...]:
    """Return a threadgroup shape of positive sizes with at most 1024 threads.

    ``axes`` fixes the number of sizes; ``None`` accepts 1 or 2. 1024 is
    Metal's documented ceiling for one threadgroup; a compiled pipeline can
    report less, which admission checks later.
    """

    expected = (1, 2) if axes is None else (axes,)
    if type(value) is not tuple or len(value) not in expected:
        count = " or ".join(str(item) for item in expected)
        raise MetalLaunchSweepError(f"{label} must be a tuple of {count} sizes")
    for axis, size in zip("xy", value):
        require_integer(size, f"{label} {axis}", 1, MAX_THREADGROUP_THREADS)
    if prod(value) > MAX_THREADGROUP_THREADS:
        raise MetalLaunchSweepError(
            f"{label} holds {prod(value)} threads; Metal allows at most "
            f"{MAX_THREADGROUP_THREADS} in one threadgroup"
        )
    return value


def shape_label(group: tuple[int, ...]) -> str:
    """Spell a shape as the report and the CLI print it: ``16x16`` or ``256``."""

    return "x".join(str(size) for size in group)


@dataclass(frozen=True)
class SimdFootprint:
    """How one threadgroup splits into SIMD groups under the linear layout.

    ``rows`` and ``columns`` are the most distinct y and x positions that any
    one SIMD group of the threadgroup spans.
    """

    simd_groups: int
    idle_lanes: int
    rows: int
    columns: int

    def to_mapping(self) -> dict[str, int]:
        """Return the footprint as a report records it."""

        return {
            "simd_groups": self.simd_groups,
            "idle_lanes": self.idle_lanes,
            "rows_per_simd_group": self.rows,
            "columns_per_simd_group": self.columns,
        }


def simd_footprint(group: tuple[int, ...], width: int) -> SimdFootprint:
    """Split a threadgroup into SIMD groups of ``width`` consecutive linear indices.

    A 1D group is one row. The last SIMD group of a group that is not a
    whole multiple of the width holds idle lanes.
    """

    shape = group_shape(group)
    lanes = require_integer(width, "thread_execution_width", 1, MAX_THREADGROUP_THREADS)
    columns_per_row = shape[0]
    threads = prod(shape)
    groups = ceil_div(threads, lanes)
    rows = columns = 0
    for first in range(0, threads, lanes):
        linear = range(first, min(first + lanes, threads))
        rows = max(rows, len({index // columns_per_row for index in linear}))
        columns = max(columns, len({index % columns_per_row for index in linear}))
    return SimdFootprint(
        simd_groups=groups,
        idle_lanes=groups * lanes - threads,
        rows=rows,
        columns=columns,
    )


@dataclass(frozen=True)
class ShapeAdmission:
    """The verdict of one pipeline on one threadgroup shape, with its reason."""

    group_size: tuple[int, ...]
    limits: PipelineLimits
    requires_whole_simd_groups: bool

    def __post_init__(self) -> None:
        group_shape(self.group_size)
        if not isinstance(self.limits, PipelineLimits):
            raise TypeError("limits must be PipelineLimits")
        if type(self.requires_whole_simd_groups) is not bool:
            raise TypeError("requires_whole_simd_groups must be a bool")

    @property
    def threads_per_group(self) -> int:
        """Threads in one threadgroup: CUDA's threads per block."""

        return prod(self.group_size)

    @property
    def reason(self) -> str | None:
        """The first rule the shape breaks, or ``None`` when the pipeline admits it."""

        if self.threads_per_group > self.limits.max_threads_per_threadgroup:
            return EXCEEDS_MAXIMUM
        width = self.limits.thread_execution_width
        if self.requires_whole_simd_groups and self.threads_per_group % width:
            return NOT_WHOLE_SIMD_GROUPS
        return None

    @property
    def admitted(self) -> bool:
        """Whether the shape may launch on this pipeline."""

        return self.reason is None

    @property
    def footprint(self) -> SimdFootprint:
        """The SIMD-group split of the shape at this pipeline's execution width."""

        return simd_footprint(self.group_size, self.limits.thread_execution_width)

    def to_mapping(self) -> dict[str, object]:
        """Return the verdict with the arithmetic behind it."""

        return {
            "group_size": list(self.group_size),
            "threads_per_group": self.threads_per_group,
            "admitted": self.admitted,
            "reason": self.reason,
            **self.footprint.to_mapping(),
        }


def admit_shapes(
    shapes: tuple[tuple[int, ...], ...],
    limits: PipelineLimits,
    *,
    requires_whole_simd_groups: bool,
) -> tuple[ShapeAdmission, ...]:
    """Check every shape of one sweep against the pipeline of its kernel."""

    if type(shapes) is not tuple or not 1 <= len(shapes) <= MAX_SHAPES:
        raise MetalLaunchSweepError(f"a sweep holds 1 to {MAX_SHAPES} shapes")
    if len(set(shapes)) != len(shapes):
        raise MetalLaunchSweepError("a sweep must not repeat a shape")
    return tuple(
        ShapeAdmission(shape, limits, requires_whole_simd_groups) for shape in shapes
    )


def matmul_launch(m: int, n: int, group: tuple[int, int]) -> LaunchGeometry:
    """Size the matmul grid with x over the n output columns and y over the m rows.

    Consecutive x threads compute consecutive columns of one output row, the
    mapping of the naive CUDA matmul that reads B along its rows.
    """

    rows = require_integer(m, "m", 1, MAX_MATRIX_SIDE)
    columns = require_integer(n, "n", 1, MAX_MATRIX_SIDE)
    return LaunchGeometry(extent=(columns, rows), group_size=group_shape(group, 2))


def checked_launch(
    admission: ShapeAdmission, geometry: LaunchGeometry
) -> CheckedLaunch:
    """Bind an admitted shape to a grid; a rejected shape never gets a launch."""

    if not isinstance(admission, ShapeAdmission) or not isinstance(
        geometry, LaunchGeometry
    ):
        raise TypeError("checked_launch needs a ShapeAdmission and a LaunchGeometry")
    if not admission.admitted:
        raise MetalLaunchSweepError(
            f"shape {shape_label(admission.group_size)} was rejected: "
            f"{admission.reason}"
        )
    if geometry.group_size != admission.group_size:
        raise MetalLaunchSweepError("the grid must use the admitted shape")
    return CheckedLaunch(geometry, admission.limits)
