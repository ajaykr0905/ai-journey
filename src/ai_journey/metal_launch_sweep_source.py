"""Metal Shading Language source of the sweep's naive matmul and SIMD layout probe.

``torch.mps.compile_shader`` compiles this text at run time. ``matmul_naive``
computes ``C = A B`` in float32 with one thread per element of ``C``. Its
global index comes from ``thread_position_in_grid``, which equals CUDA's
``blockIdx * blockDim + threadIdx``: x runs over the ``n`` columns and y over
the ``m`` rows, the mapping of the naive CUDA matmul. Adjacent x threads read
adjacent elements of one row of ``B`` and the same element of ``A``. The grid
is whole threadgroups, so the kernel guards both axes. The kernel has no
threadgroup memory and no SIMD-group functions, so every threadgroup shape
runs the same per-thread arithmetic. Its outputs must then be bit-identical
across shapes, which the correctness checks require.

``simd_layout_probe`` runs as one threadgroup. Each thread writes, at its
linear index in the threadgroup, the SIMD group and lane it runs in, its x
and y position, and the SIMD width it sees. That record is an observation of
how this host splits a threadgroup, not an assumption about it.

The 1D sweep times the Day 73 ``vector_add`` kernel, compiled from the Day 73
source by the Day 73 runtime. This source holds only what that one lacks.
``matmul_thread`` is the matmul kernel written per thread for the Day 73
launch simulator, with the same index and guard statements.
"""

from __future__ import annotations

from hashlib import sha256

from .kernel_launch_buffer import DeviceBuffer, to_float32
from .kernel_launch_simulator import ThreadContext

MATMUL = "matmul_naive"
SIMD_LAYOUT_PROBE = "simd_layout_probe"
SWEEP_KERNELS = (MATMUL, SIMD_LAYOUT_PROBE)
# The words that the probe writes for each thread, in order.
LAYOUT_FIELDS = ("simd_group", "lane", "x", "y", "simd_width")

SWEEP_SOURCE = f"""#include <metal_stdlib>
using namespace metal;

// One thread per element of C = A B; every matrix is row-major float32.
// x runs over the n columns and y over the m rows, as in the naive CUDA
// matmul: col = blockIdx.x * blockDim.x + threadIdx.x. The grid is whole
// threadgroups, so the guard stops the padding threads on both axes.
kernel void {MATMUL}(device float* c [[buffer(0)]],
                         device const float* a [[buffer(1)]],
                         device const float* b [[buffer(2)]],
                         constant uint& m [[buffer(3)]],
                         constant uint& n [[buffer(4)]],
                         constant uint& k [[buffer(5)]],
                         uint2 position [[thread_position_in_grid]]) {{
    uint column = position.x;
    uint row = position.y;
    if (row < m && column < n) {{
        float sum = 0.0f;
        for (uint i = 0; i < k; ++i) {{
            sum += a[row * k + i] * b[i * n + column];
        }}
        c[row * n + column] = sum;
    }}
}}

// Run as one threadgroup. At its linear index, each thread records its SIMD
// group, its lane, its x and y position and the SIMD width it sees.
kernel void {SIMD_LAYOUT_PROBE}(device uint* out [[buffer(0)]],
                              uint linear [[thread_index_in_threadgroup]],
                              uint2 position [[thread_position_in_threadgroup]],
                              uint simd_group [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint width [[threads_per_simdgroup]]) {{
    out[{len(LAYOUT_FIELDS)} * linear] = simd_group;
    out[{len(LAYOUT_FIELDS)} * linear + 1] = lane;
    out[{len(LAYOUT_FIELDS)} * linear + 2] = position.x;
    out[{len(LAYOUT_FIELDS)} * linear + 3] = position.y;
    out[{len(LAYOUT_FIELDS)} * linear + 4] = width;
}}
"""

# Buffer bindings in order; the scalars m, n and k are 32-bit constant uint&.
MATMUL_BUFFERS = ("c", "a", "b", "m", "n", "k")
MATMUL_SCALAR_BUFFERS = (3, 4, 5)
PROBE_BUFFERS = ("out",)


def matmul_arg_casts() -> dict[int, str]:
    """Pass m, n and k as 32 bits; PyTorch would otherwise pass 64."""

    return {index: "int32" for index in MATMUL_SCALAR_BUFFERS}


def kernel_bindings() -> list[dict[str, object]]:
    """Return the bindings of the sweep kernels as a report records them."""

    return [
        {
            "name": MATMUL,
            "axes": 2,
            "buffers": list(MATMUL_BUFFERS),
            "scalar_buffers": list(MATMUL_SCALAR_BUFFERS),
        },
        {
            "name": SIMD_LAYOUT_PROBE,
            "axes": 2,
            "buffers": list(PROBE_BUFFERS),
            "scalar_buffers": [],
        },
    ]


def source_sha256() -> str:
    """Hash the exact source text that ``compile_shader`` receives."""

    return sha256(SWEEP_SOURCE.encode("ascii")).hexdigest()


def matmul_thread(
    t: ThreadContext,
    a: DeviceBuffer,
    b: DeviceBuffer,
    c: DeviceBuffer,
    m: int,
    n: int,
    k: int,
) -> None:
    """Run one thread of ``matmul_naive`` in the Day 73 launch simulator.

    The index and guard statements are those of the Metal source. Each
    product and sum rounds to float32, as two separate float32 operations.
    """

    column = t.block_idx.x * t.block_dim.x + t.thread_idx.x
    row = t.block_idx.y * t.block_dim.y + t.thread_idx.y
    if row < m and column < n:
        total = 0.0
        for i in range(k):
            total = to_float32(total + to_float32(a[row * k + i] * b[i * n + column]))
        c[row * n + column] = total
