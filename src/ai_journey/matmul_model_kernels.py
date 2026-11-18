"""Per-thread naive matmul kernels for the Day 73 launch simulator.

``naive_matmul_kernel`` is the Day 74 exercise kernel in the per-thread style
of the Day 73 kernels. A thread takes ``row`` from y and ``col`` from x, and a
guard on both axes keeps it inside ``C``. The thread then sums ``A[row, i] *
B[i, col]`` for ``i`` from 0 to ``k - 1`` in the stated order of
``matmul_model_reference`` and stores one element of ``C``.

``naive_matmul_rows_kernel`` swaps the axes: ``row`` from x and ``col`` from
y. Every element of ``C`` still gets the same sum in the same order, so both
kernels give the same bits, but the 32 threads of one warp then read other
addresses. ``matmul_model_access`` counts that difference.

No thread writes an element that another thread reads or writes, so the
result cannot depend on the launch order. The run helper sizes the grid with
``cdiv`` on each axis, as the Day 73 run helpers do.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .kernel_launch_buffer import DeviceBuffer, to_float32
from .kernel_launch_config import Dim3, LaunchConfig, launch_2d
from .kernel_launch_kernels import KernelRun
from .kernel_launch_simulator import LaunchOrder, ThreadContext, launch
from .matmul_model_reference import MatmulModelError, require_shape
from .memory_access_tiling import MatmulShape

X_COLUMNS = "x_columns"
X_ROWS = "x_rows"
MAPPINGS = (X_COLUMNS, X_ROWS)


def _guarded_dot(
    a: DeviceBuffer,
    b: DeviceBuffer,
    c: DeviceBuffer,
    row: int,
    col: int,
    m: int,
    n: int,
    k: int,
) -> None:
    """Store ``C[row, col]`` in the stated order when the thread is inside C."""

    if row < m and col < n:
        total = 0.0
        for i in range(k):
            total = to_float32(total + to_float32(a[row * k + i] * b[i * n + col]))
        c[row * n + col] = total


def naive_matmul_kernel(
    t: ThreadContext,
    a: DeviceBuffer,
    b: DeviceBuffer,
    c: DeviceBuffer,
    m: int,
    n: int,
    k: int,
) -> None:
    """Run one thread of ``C = A @ B`` with row from y and col from x."""

    row = t.block_idx.y * t.block_dim.y + t.thread_idx.y
    col = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    _guarded_dot(a, b, c, row, col, m, n, k)


def naive_matmul_rows_kernel(
    t: ThreadContext,
    a: DeviceBuffer,
    b: DeviceBuffer,
    c: DeviceBuffer,
    m: int,
    n: int,
    k: int,
) -> None:
    """Run one thread with the axes swapped: row from x and col from y."""

    row = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    col = t.block_idx.y * t.block_dim.y + t.thread_idx.y
    _guarded_dot(a, b, c, row, col, m, n, k)


KERNELS: Mapping[str, object] = MappingProxyType(
    {X_COLUMNS: naive_matmul_kernel, X_ROWS: naive_matmul_rows_kernel}
)


def require_mapping(mapping: object) -> str:
    """Return a thread mapping name: ``x_columns`` or ``x_rows``."""

    if type(mapping) is not str or mapping not in MAPPINGS:
        raise MatmulModelError(f"mapping must be one of {MAPPINGS}")
    return mapping


def thread_extent(shape: MatmulShape, mapping: str) -> Dim3:
    """Return the (x, y) extent that the threads of a mapping must cover."""

    checked = require_shape(shape)
    if require_mapping(mapping) == X_COLUMNS:
        return Dim3(checked.n, checked.m)
    return Dim3(checked.m, checked.n)


def matmul_launch(
    shape: MatmulShape, block: tuple[int, int], mapping: str
) -> LaunchConfig:
    """Size the smallest grid of ``block`` that gives each element a thread."""

    if type(block) is not tuple or len(block) != 2:
        raise MatmulModelError("block must be a (block_x, block_y) tuple")
    extent = thread_extent(shape, mapping)
    return launch_2d(extent.x, extent.y, *block)


def run_naive_matmul(
    a: Sequence[float],
    b: Sequence[float],
    shape: MatmulShape,
    *,
    block: tuple[int, int],
    mapping: str = X_COLUMNS,
    order: LaunchOrder | None = None,
) -> KernelRun:
    """Launch one naive matmul in ``order`` and copy ``C`` back to the host."""

    config = matmul_launch(shape, block, mapping)
    if len(a) != shape.m * shape.k or len(b) != shape.k * shape.n:
        raise MatmulModelError("A needs m * k values and B needs k * n values")
    c = DeviceBuffer.zeros("c", "float32", shape.m * shape.n)
    result = launch(
        KERNELS[mapping],
        config,
        DeviceBuffer("a", "float32", a),
        DeviceBuffer("b", "float32", b),
        c,
        shape.m,
        shape.n,
        shape.k,
        order=order,
    )
    return KernelRun(
        kernel=f"naive_matmul_{mapping}",
        extent=thread_extent(shape, mapping),
        output_buffer="c",
        result=result,
        output=c.to_host(),
        output_sha256=c.sha256(),
    )
