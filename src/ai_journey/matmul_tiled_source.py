"""Metal Shading Language source of the tiled float32 matmul, one kernel per tile.

A tiled kernel gives one T x T threadgroup one T x T tile of ``C = A @ B``,
with x over the columns of C, as in the faster Day 74 mapping. It walks K in
phases of T. In each phase every thread copies one element of A and one of B
into two static ``threadgroup`` arrays, Metal's counterpart of CUDA
``__shared__`` memory, and the group waits at a barrier. Each thread then
reads T staged elements of each array, so one global load serves T outputs,
and the group waits again before the next phase overwrites the tiles.

Three rules keep the arithmetic equal to ``fma_replay``:

- The inner loop of the phase that starts at ``start`` runs
  ``i < min(T, k - start)``, so each output takes exactly K steps of
  ``LOOP_STATEMENT`` in increasing k from 0. A zero-filled slot of a K tail
  is never read: one more ``fma(0, 0, sum)`` would turn a sum of -0.0 into
  +0.0, so bit equality must not depend on it.
- A slot past M (in A) or past N (in B) holds 0 and feeds only the outputs
  of padding threads, which never store.
- Only the store is guarded, so every thread of a group reaches both
  barriers of every phase, as ``threadgroup_barrier`` requires.

A ``compile_shader`` launch takes only ``threads``, ``group_size`` and
``arg_casts``, so a launch cannot size threadgroup memory. Each tile size is
therefore its own kernel with arrays of a fixed size, ``2 * T * T * 4`` bytes.
The buffers and the 32-bit scalars in slots 3 to 5 are bound exactly as in
the Day 74 kernels.
"""

from __future__ import annotations

from .kernel_metal_geometry import require_integer
from .matmul_metal_mapping import FLOAT32_BYTES, MatmulMetalError
from .memory_access_tiling import MAX_MATMUL_TILE

TILE_SIZES = (8, 16, 32)
HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"
LOOP_STATEMENT = "sum = fma(a_tile[local.y][i], b_tile[i][local.x], sum);"
BARRIER = "threadgroup_barrier(mem_flags::mem_threadgroup);"
STORE_GUARD = "if (row < m && column < n) {"


class MatmulTiledError(MatmulMetalError):
    """Raised when a tiled matmul control, launch, check or report is invalid."""


def require_tile(tile: object) -> int:
    """Return a tile size of 1 to 32.

    One thread runs per element of a T x T tile, and Metal allows at most
    1024 = 32 * 32 threads in one threadgroup.
    """

    return require_integer(tile, "tile", 1, MAX_MATMUL_TILE)


def tiled_kernel_name(tile: int) -> str:
    """Name the kernel of one tile size: ``matmul_tiled_16`` for T = 16."""

    return f"matmul_tiled_{require_tile(tile)}"


def threadgroup_bytes(tile: int) -> int:
    """Bytes of the two static float32 tiles: ``2 * T * T * 4``."""

    size = require_tile(tile)
    return 2 * size * size * FLOAT32_BYTES


def tiled_kernel_source(tile: int) -> str:
    """Return the kernel text of one tile size, without the library header."""

    size = require_tile(tile)
    name = tiled_kernel_name(size)
    signature = f"kernel void {name}("
    indent = " " * len(signature)
    parameters = f",\n{indent}".join(
        (
            "device float* c [[buffer(0)]]",
            "device const float* a [[buffer(1)]]",
            "device const float* b [[buffer(2)]]",
            "constant uint& m [[buffer(3)]]",
            "constant uint& n [[buffer(4)]]",
            "constant uint& k [[buffer(5)]]",
            "uint2 position [[thread_position_in_grid]]",
            "uint2 local [[thread_position_in_threadgroup]]",
        )
    )
    lines = (
        f"// One {size} x {size} threadgroup per {size} x {size} tile of C = A @ B; "
        "all three",
        "// are row-major float32. Each phase stages one tile of A and one of B,",
        "// zero-filled past the matrices, and the inner loop stops at the end",
        "// of K, so every output takes K fma steps in increasing k.",
        f"{signature}{parameters}) {{",
        f"    threadgroup float a_tile[{size}][{size}];",
        f"    threadgroup float b_tile[{size}][{size}];",
        "    uint column = position.x;",
        "    uint row = position.y;",
        "    float sum = 0.0f;",
        f"    for (uint start = 0; start < k; start += {size}) {{",
        "        uint a_column = start + local.x;",
        "        uint b_row = start + local.y;",
        "        a_tile[local.y][local.x] =",
        "            row < m && a_column < k ? a[row * k + a_column] : 0.0f;",
        "        b_tile[local.y][local.x] =",
        "            b_row < k && column < n ? b[b_row * n + column] : 0.0f;",
        f"        {BARRIER}",
        f"        uint steps = min({size}u, k - start);",
        "        for (uint i = 0; i < steps; ++i) {",
        f"            {LOOP_STATEMENT}",
        "        }",
        f"        {BARRIER}",
        "    }",
        f"    {STORE_GUARD}",
        "        c[row * n + column] = sum;",
        "    }",
        "}",
    )
    return "\n".join(lines) + "\n"
