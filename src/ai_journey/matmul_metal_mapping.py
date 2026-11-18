"""Two one-thread-per-output mappings of a naive matmul onto a 2D Metal grid.

The lab computes ``C[M, N] = A[M, K] @ B[K, N]`` in float32 with every matrix
row-major, and gives each thread one element of C. The grid's x axis is the
fastest-varying axis, so threads that are consecutive in x run side by side.
The two mappings differ only in the matrix axis that x walks:

- ``x_columns``: thread ``(x, y)`` computes ``C[y, x]``. At each step ``k``,
  consecutive threads read consecutive elements of one row of B, all read the
  same element of A, and they write consecutive elements of C.
- ``x_rows``: thread ``(x, y)`` computes ``C[x, y]``. Consecutive threads read
  elements of A that lie K floats apart, all read the same element of B, and
  they write elements of C that lie N floats apart.

Both mappings run the same loop over k for each output, so they compute the
same bits; only the addresses that neighboring threads touch together
change. ``AccessSteps`` derives those address steps from the index formulas,
so a report can state them without a profiler. Launch geometry reuses the
Day 73 ``LaunchGeometry``: whole threadgroups by ceiling division on both
axes, with a bounds guard in the kernel for the padding threads.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kernel_metal_geometry import KernelMetalError, LaunchGeometry, require_integer
from .memory_access_tiling import MatmulShape

X_COLUMNS = "x_columns"
X_ROWS = "x_rows"
MAPPINGS = (X_COLUMNS, X_ROWS)
FLOAT32_BYTES = 4
# With at most 8192 per dimension, each matrix holds at most 2**26 float32
# elements (256 MiB), and every index the kernels form in 32-bit uint, such
# as row * k + i, fits.
MAX_DIMENSION = 8192


class MatmulMetalError(KernelMetalError):
    """Raised when a Metal matmul control, launch, check or report is invalid."""


def matmul_shape(m: object, n: object, k: object) -> MatmulShape:
    """Return the shape ``C[m, n] = A[m, k] @ B[k, n]`` inside the lab's bounds.

    The bound on each dimension keeps every index that the kernels form in
    32-bit unsigned integers exact.
    """

    return MatmulShape(
        require_integer(m, "m", 1, MAX_DIMENSION),
        require_integer(n, "n", 1, MAX_DIMENSION),
        require_integer(k, "k", 1, MAX_DIMENSION),
    )


def require_mapping(mapping: object) -> str:
    """Return a known mapping name, so a typo cannot select a default."""

    if mapping not in MAPPINGS:
        raise MatmulMetalError(f"mapping must be one of {', '.join(MAPPINGS)}")
    return mapping


def _shape(shape: object) -> MatmulShape:
    if not isinstance(shape, MatmulShape):
        raise TypeError("shape must be a MatmulShape")
    return matmul_shape(shape.m, shape.n, shape.k)


def grid_extent(mapping: str, shape: MatmulShape) -> tuple[int, int]:
    """Return the data extent ``(x, y)``: ``(N, M)`` or, for x over rows, ``(M, N)``."""

    checked = _shape(shape)
    if require_mapping(mapping) == X_COLUMNS:
        return checked.n, checked.m
    return checked.m, checked.n


def output_element(mapping: str, x: int, y: int) -> tuple[int, int]:
    """Return the ``(row, column)`` of C that grid thread ``(x, y)`` computes."""

    return (y, x) if require_mapping(mapping) == X_COLUMNS else (x, y)


def matmul_launch(
    mapping: str, shape: MatmulShape, group_size: tuple[int, int]
) -> LaunchGeometry:
    """Size one launch with one thread per element of C, in ``(x, y)`` groups.

    The same group size serves both mappings, so a timing difference between
    them comes from the mapping alone.
    """

    if type(group_size) is not tuple or len(group_size) != 2:
        raise MatmulMetalError("a matmul group_size must be an (x, y) tuple")
    return LaunchGeometry(extent=grid_extent(mapping, shape), group_size=group_size)


@dataclass(frozen=True)
class AccessSteps:
    """How far each matrix address moves when a thread's x or y grows by one.

    Steps count float32 elements of the row-major matrices. A step of 0 means
    neighboring threads read one shared element; a step of 1 means they read
    consecutive elements. Within one thread, every k step moves A by 1 and B
    by N in both mappings.
    """

    mapping: str
    a: tuple[int, int]
    b: tuple[int, int]
    c: tuple[int, int]

    def to_mapping(self) -> dict[str, object]:
        """Return the steps in bytes, as a report records them."""

        def steps(pair: tuple[int, int]) -> dict[str, int]:
            x_step, y_step = pair
            return {"x": x_step * FLOAT32_BYTES, "y": y_step * FLOAT32_BYTES}

        return {
            "mapping": self.mapping,
            "a_step_bytes": steps(self.a),
            "b_step_bytes": steps(self.b),
            "c_step_bytes": steps(self.c),
        }


def access_steps(mapping: str, shape: MatmulShape) -> AccessSteps:
    """Derive one mapping's address steps from the kernels' index formulas.

    The kernels read ``A[row * K + k]`` and ``B[k * N + column]`` and write
    ``C[row * N + column]``; each mapping fixes which of row and column is x.
    """

    checked = _shape(shape)
    inner, columns = checked.k, checked.n
    if require_mapping(mapping) == X_COLUMNS:
        # row = y and column = x.
        return AccessSteps(mapping, a=(0, inner), b=(1, 0), c=(1, columns))
    # row = x and column = y.
    return AccessSteps(mapping, a=(inner, 0), b=(0, 1), c=(columns, 1))
