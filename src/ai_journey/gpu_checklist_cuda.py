"""Pinned CUDA C++ text of a thread-coarsened and a 2D register-blocked matmul.

Day 77 applies two items of a CUDA performance checklist to the Day 76 tiled
kernel. Both kernels keep the Day 76 arithmetic, so a later NVIDIA run can
check each output against ``matmul_metal_reference.fma_replay`` bit for bit:
each element of C starts at +0.0, and ``__fmaf_rn`` adds its terms once each,
in increasing k, with one rounding per step. Each inner loop stops at the end
of k, so no padded slot enters a sum. Both kernels guard every global load
with a zero fill and every store, and no barrier is inside a guard, so every
thread of a block reaches each barrier.

* Thread coarsening (Hwu, Kirk and El Hajj, PMPP 4th ed., section 6.3): a
  T x T block computes a T x (C * T) tile of C, and each thread computes C
  elements of one row, T columns apart. A phase loads its tile of A once and
  uses it for C tiles of B, so the loads of A fall by about a factor C. Each
  B tile has its own two barriers.
* 2D register blocking: a block computes a BM x BN tile of C, and each thread
  computes a TM x TN micro-tile in registers. In each step of k, a thread
  reads TM values of A and TN values of B from shared memory and does
  TM * TN fma operations. Each thread stages 4 consecutive floats of A and 4
  of B in each phase. It uses one ``float4`` load only if the row length is a
  multiple of 4 and the base pointer is 16-byte aligned. Then the element
  offset is a multiple of 4, so the address is 16-byte aligned, and all 4
  floats are inside the row. Otherwise it uses 4 guarded scalar loads.

This lab never compiles, links or runs these texts, and this host has no
NVIDIA GPU, CUDA toolkit or ``nvcc``. The texts are source for a later run on
an NVIDIA GPU, and each text has a SHA-256 digest, so that run can show that
it compiled the checked text. The texts use no ``#pragma unroll``: they
expect that ``nvcc`` unrolls the loops with a constant trip count, so the
accumulator arrays stay in registers. A later run must check that with the
register and local memory counts of ``ptxas``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gpu_tiled_cuda import FLOAT_BYTES, TILES, GpuTiledError, cublas_source
from .gpu_tiled_cuda import require_shape as _require_tiled_shape
from .gpu_tiled_cuda import source_sha256 as _tiled_sha256
from .kernel_launch_config import Dim3, LaunchConfig, ceil_div
from .kernel_launch_cuda import _CDIV
from .matmul_model_occupancy import ARCHITECTURE
from .memory_access_tiling import MatmulShape

COARSENING_FACTORS = (2, 4, 8)
MICRO_TILE_SIZES = (2, 4, 8)
VECTOR_WIDTH = 4
VECTOR_BYTES = VECTOR_WIDTH * FLOAT_BYTES
MAX_STATIC_SHARED_BYTES = ARCHITECTURE.shared_memory_per_block_bytes
CUBLAS_FILE = "sgemm_row_major.cu"
_MAX_BLOCK_TILE = 256


class GpuChecklistError(GpuTiledError):
    """Raised when a Day 77 checklist kernel, text or run breaks its contract."""


def require_shape(shape: object) -> MatmulShape:
    """Return a shape whose int indices stay exact in both kernels.

    The Day 76 bound of 8192 per dimension applies: the largest index that a
    kernel forms, such as ``row * k + a_col``, stays below 2**26.
    """

    try:
        return _require_tiled_shape(shape)
    except GpuTiledError as error:
        raise GpuChecklistError(str(error)) from error


def source_sha256(text: str) -> str:
    """Return the SHA-256 of an ASCII source text."""

    try:
        return _tiled_sha256(text)
    except GpuTiledError as error:
        raise GpuChecklistError(str(error)) from error


def _require_choice(value: object, name: str, choices: tuple[int, ...]) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value not in choices:
        raise GpuChecklistError(f"{name} must be one of {choices}")
    return value


def _launch(
    shape: MatmulShape, block: tuple[int, int], tile: tuple[int, int]
) -> LaunchConfig:
    """Return the smallest grid of ``block`` that covers C in ``tile`` units.

    ``tile`` is the (columns, rows) of C that one block computes. Python
    integers give the exact ceiling, with no C wrap, so a check can compare
    a launcher text with this model.
    """

    checked = require_shape(shape)
    columns, rows = tile
    return LaunchConfig(
        Dim3(ceil_div(checked.n, columns), ceil_div(checked.m, rows)),
        Dim3(*block),
    )


@dataclass(frozen=True)
class CoarsenedKernel:
    """One thread-coarsened kernel: a T x T block and C outputs per thread."""

    tile: int
    coarsening: int

    def __post_init__(self) -> None:
        _require_choice(self.tile, "tile", TILES)
        _require_choice(self.coarsening, "coarsening", COARSENING_FACTORS)

    @property
    def name(self) -> str:
        return f"coarsened_matmul_t{self.tile}_c{self.coarsening}"

    @property
    def kernel_name(self) -> str:
        return f"{self.name}_kernel"

    @property
    def launcher_name(self) -> str:
        return f"launch_{self.name}"

    @property
    def width(self) -> int:
        """Return the columns of C that one block computes: C * T."""

        return self.tile * self.coarsening

    @property
    def block(self) -> tuple[int, int]:
        return (self.tile, self.tile)

    @property
    def threads_per_block(self) -> int:
        return self.tile * self.tile

    @property
    def outputs_per_thread(self) -> int:
        return self.coarsening

    @property
    def shared_memory_bytes(self) -> int:
        """Return the static shared memory of one block: two T x T float tiles."""

        return 2 * self.tile * self.tile * FLOAT_BYTES

    def launch(self, shape: MatmulShape) -> LaunchConfig:
        """Return the grid of ``cdiv(n, C * T)`` by ``cdiv(m, T)`` blocks."""

        return _launch(shape, self.block, (self.width, self.tile))

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": "coarsened",
            "name": self.name,
            "tile": self.tile,
            "coarsening": self.coarsening,
            "block": list(self.block),
            "output_tile": [self.tile, self.width],
            "threads_per_block": self.threads_per_block,
            "shared_memory_bytes": self.shared_memory_bytes,
        }


@dataclass(frozen=True)
class BlockedKernel:
    """One 2D register-blocked kernel: BM x BN of C per block, TM x TN per thread.

    Each thread stages exactly one group of 4 floats of A and one of B in
    each phase, so ``BM * BK`` and ``BK * BN`` must both be 4 times the
    threads of a block. With ``BK`` and ``BN`` multiples of 4, each group
    starts at a multiple of 4 inside its row. The rule also bounds the two
    tiles at 32 bytes per thread, at most 32 KB, inside the 48 KB of static
    shared memory.
    """

    block_m: int
    block_n: int
    block_k: int
    thread_m: int
    thread_n: int

    def __post_init__(self) -> None:
        for name in ("block_m", "block_n", "block_k"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if not VECTOR_WIDTH <= value <= _MAX_BLOCK_TILE:
                raise GpuChecklistError(
                    f"{name} must be between {VECTOR_WIDTH} and {_MAX_BLOCK_TILE}"
                )
        _require_choice(self.thread_m, "thread_m", MICRO_TILE_SIZES)
        _require_choice(self.thread_n, "thread_n", MICRO_TILE_SIZES)
        if self.block_m % self.thread_m or self.block_n % self.thread_n:
            raise GpuChecklistError("each micro-tile must divide its block tile")
        threads = self.threads_per_block
        if threads % ARCHITECTURE.warp_size or threads > (
            ARCHITECTURE.max_threads_per_block
        ):
            raise GpuChecklistError(
                "a block must hold whole warps and at most "
                f"{ARCHITECTURE.max_threads_per_block} threads"
            )
        if self.block_k % VECTOR_WIDTH or self.block_n % VECTOR_WIDTH:
            raise GpuChecklistError("block_k and block_n must be multiples of 4")
        if not (
            self.block_m * self.block_k
            == VECTOR_WIDTH * threads
            == self.block_k * self.block_n
        ):
            raise GpuChecklistError(
                "each thread must stage one group of 4 floats of A and one of B "
                "per phase: block_m * block_k and block_k * block_n must both be "
                "4 times the threads of a block"
            )

    @property
    def name(self) -> str:
        return (
            f"blocked_matmul_m{self.block_m}_n{self.block_n}_k{self.block_k}"
            f"_t{self.thread_m}x{self.thread_n}"
        )

    @property
    def kernel_name(self) -> str:
        return f"{self.name}_kernel"

    @property
    def launcher_name(self) -> str:
        return f"launch_{self.name}"

    @property
    def block(self) -> tuple[int, int]:
        """Return (x, y): x runs over micro-tile columns, y over micro-tile rows."""

        return (self.block_n // self.thread_n, self.block_m // self.thread_m)

    @property
    def threads_per_block(self) -> int:
        x, y = self.block
        return x * y

    @property
    def outputs_per_thread(self) -> int:
        return self.thread_m * self.thread_n

    @property
    def a_groups_per_row(self) -> int:
        """Return the groups of 4 floats in one row of the A tile."""

        return self.block_k // VECTOR_WIDTH

    @property
    def b_groups_per_row(self) -> int:
        """Return the groups of 4 floats in one row of the B tile."""

        return self.block_n // VECTOR_WIDTH

    @property
    def shared_memory_bytes(self) -> int:
        """Return the static shared memory: a BK x BM tile and a BK x BN tile."""

        return (self.block_k * self.block_m + self.block_k * self.block_n) * (
            FLOAT_BYTES
        )

    def launch(self, shape: MatmulShape) -> LaunchConfig:
        """Return the grid of ``cdiv(n, BN)`` by ``cdiv(m, BM)`` blocks."""

        return _launch(shape, self.block, (self.block_n, self.block_m))

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": "blocked",
            "name": self.name,
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "thread_m": self.thread_m,
            "thread_n": self.thread_n,
            "block": list(self.block),
            "output_tile": [self.block_m, self.block_n],
            "threads_per_block": self.threads_per_block,
            "shared_memory_bytes": self.shared_memory_bytes,
        }


ChecklistKernel = CoarsenedKernel | BlockedKernel
PINNED_COARSENED = (
    CoarsenedKernel(tile=16, coarsening=2),
    CoarsenedKernel(tile=16, coarsening=4),
    CoarsenedKernel(tile=32, coarsening=2),
)
PINNED_BLOCKED = (
    BlockedKernel(block_m=64, block_n=64, block_k=16, thread_m=4, thread_n=4),
    BlockedKernel(block_m=128, block_n=128, block_k=8, thread_m=8, thread_n=8),
)
PINNED_KERNELS: tuple[ChecklistKernel, ...] = (*PINNED_COARSENED, *PINNED_BLOCKED)


def require_kernel(kernel: object) -> ChecklistKernel:
    """Return a coarsened or a blocked kernel; anything else is refused."""

    if type(kernel) not in (CoarsenedKernel, BlockedKernel):
        raise TypeError("kernel must be CoarsenedKernel or BlockedKernel")
    return kernel


_HEADER = """\
// CUDA C++ source generated by ai_journey.gpu_checklist_cuda.
// The lab that generated this text did not compile or run it.
// It is source for a later run on an NVIDIA GPU.
"""
_SIGNATURE = "const float* a, const float* b, float* c, int m, int n, int k"


def coarsened_cuda_source(kernel: CoarsenedKernel) -> str:
    """Return the pinned CUDA C++ text of one thread-coarsened kernel."""

    if type(kernel) is not CoarsenedKernel:
        raise TypeError("kernel must be CoarsenedKernel")
    t, factor, width = kernel.tile, kernel.coarsening, kernel.width
    return f"""\
{_HEADER}\
// Thread coarsening: one {t} x {t} block computes a {t} x {width} tile of C,
// and each thread computes {factor} elements of one row of C, {t} columns
// apart. In each phase, the block loads one {t} x {t} tile of A once and uses
// it for {factor} tiles of B. A barrier follows each load of a B tile, and a
// second barrier follows each inner loop. The inner loop stops at
// min({t}, k - phase * {t}), and __fmaf_rn rounds each step once, in increasing
// k for each element of C. Only the loads and the stores are guarded, so
// every thread reaches each barrier.

{_CDIV} {{
    return (a + b - 1) / b;
}}

__global__ void {kernel.kernel_name}(
    {_SIGNATURE}) {{
    __shared__ float As[{t}][{t}];
    __shared__ float Bs[{t}][{t}];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * {t} + ty;
    int col0 = blockIdx.x * {width} + tx;
    float sum[{factor}];
    for (int j = 0; j < {factor}; ++j) {{
        sum[j] = 0.0f;
    }}
    for (int phase = 0; phase * {t} < k; ++phase) {{
        int a_col = phase * {t} + tx;
        int b_row = phase * {t} + ty;
        As[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        for (int j = 0; j < {factor}; ++j) {{
            int col = col0 + j * {t};
            Bs[ty][tx] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
            __syncthreads();
            for (int i = 0; i < min({t}, k - phase * {t}); ++i) {{
                sum[j] = __fmaf_rn(As[ty][i], Bs[i][tx], sum[j]);
            }}
            __syncthreads();
        }}
    }}
    for (int j = 0; j < {factor}; ++j) {{
        int col = col0 + j * {t};
        if (row < m && col < n) {{
            c[row * n + col] = sum[j];
        }}
    }}
}}

void {kernel.launcher_name}(
    {_SIGNATURE}) {{
    dim3 block({t}, {t});
    dim3 grid(cdiv(n, {width}), cdiv(m, {t}));
    {kernel.kernel_name}<<<grid, block>>>(a, b, c, m, n, k);
}}
"""


def blocked_cuda_source(kernel: BlockedKernel) -> str:
    """Return the pinned CUDA C++ text of one 2D register-blocked kernel."""

    if type(kernel) is not BlockedKernel:
        raise TypeError("kernel must be BlockedKernel")
    bm, bn, bk = kernel.block_m, kernel.block_n, kernel.block_k
    tm, tn = kernel.thread_m, kernel.thread_n
    x, y = kernel.block
    ga, gb = kernel.a_groups_per_row, kernel.b_groups_per_row
    zero4 = "make_float4(0.0f, 0.0f, 0.0f, 0.0f)"
    return f"""\
{_HEADER}\
// 2D register blocking: one {x} x {y} block computes a {bm} x {bn} tile of C,
// and each thread computes a {tm} x {tn} micro-tile of C in registers. In each
// phase, the block stages a {bm} x {bk} tile of A, transposed, and a {bk} x {bn}
// tile of B in shared memory. Each thread stages 4 consecutive floats of A
// and 4 of B. It uses one float4 load only if the row length is a multiple
// of 4 and the base pointer is 16-byte aligned. Then the address is 16-byte
// aligned, and all 4 floats are inside the row. Otherwise it uses 4 guarded
// scalar loads. A barrier follows the loads, and a second barrier follows the
// inner loop. The inner loop stops at min({bk}, k - phase * {bk}), and
// __fmaf_rn rounds each step once, in increasing k for each element of C.
// Only the loads and the stores are guarded, so every thread reaches each
// barrier.

{_CDIV} {{
    return (a + b - 1) / b;
}}

__global__ void {kernel.kernel_name}(
    {_SIGNATURE}) {{
    __shared__ float As[{bk}][{bm}];
    __shared__ float Bs[{bk}][{bn}];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int t = ty * {x} + tx;
    int row0 = blockIdx.y * {bm} + ty * {tm};
    int col0 = blockIdx.x * {bn} + tx * {tn};
    int a_r = t / {ga};
    int a_c = t % {ga} * 4;
    int b_r = t / {gb};
    int b_c = t % {gb} * 4;
    int a_row = blockIdx.y * {bm} + a_r;
    int b_col = blockIdx.x * {bn} + b_c;
    bool a_vector =
        k % 4 == 0 && reinterpret_cast<unsigned long long>(a) % 16 == 0;
    bool b_vector =
        n % 4 == 0 && reinterpret_cast<unsigned long long>(b) % 16 == 0;
    float sum[{tm}][{tn}];
    float a_reg[{tm}];
    float b_reg[{tn}];
    for (int mi = 0; mi < {tm}; ++mi) {{
        for (int ni = 0; ni < {tn}; ++ni) {{
            sum[mi][ni] = 0.0f;
        }}
    }}
    for (int phase = 0; phase * {bk} < k; ++phase) {{
        int a_col = phase * {bk} + a_c;
        int b_row = phase * {bk} + b_r;
        if (a_vector) {{
            float4 v = (a_row < m && a_col < k)
                ? *reinterpret_cast<const float4*>(&a[a_row * k + a_col])
                : {zero4};
            As[a_c][a_r] = v.x;
            As[a_c + 1][a_r] = v.y;
            As[a_c + 2][a_r] = v.z;
            As[a_c + 3][a_r] = v.w;
        }} else {{
            for (int q = 0; q < 4; ++q) {{
                As[a_c + q][a_r] =
                    (a_row < m && a_col + q < k) ? a[a_row * k + a_col + q] : 0.0f;
            }}
        }}
        if (b_vector) {{
            float4 v = (b_row < k && b_col < n)
                ? *reinterpret_cast<const float4*>(&b[b_row * n + b_col])
                : {zero4};
            Bs[b_r][b_c] = v.x;
            Bs[b_r][b_c + 1] = v.y;
            Bs[b_r][b_c + 2] = v.z;
            Bs[b_r][b_c + 3] = v.w;
        }} else {{
            for (int q = 0; q < 4; ++q) {{
                Bs[b_r][b_c + q] =
                    (b_row < k && b_col + q < n) ? b[b_row * n + b_col + q] : 0.0f;
            }}
        }}
        __syncthreads();
        for (int i = 0; i < min({bk}, k - phase * {bk}); ++i) {{
            for (int mi = 0; mi < {tm}; ++mi) {{
                a_reg[mi] = As[i][ty * {tm} + mi];
            }}
            for (int ni = 0; ni < {tn}; ++ni) {{
                b_reg[ni] = Bs[i][tx * {tn} + ni];
            }}
            for (int mi = 0; mi < {tm}; ++mi) {{
                for (int ni = 0; ni < {tn}; ++ni) {{
                    sum[mi][ni] = __fmaf_rn(a_reg[mi], b_reg[ni], sum[mi][ni]);
                }}
            }}
        }}
        __syncthreads();
    }}
    for (int mi = 0; mi < {tm}; ++mi) {{
        int row = row0 + mi;
        for (int ni = 0; ni < {tn}; ++ni) {{
            int col = col0 + ni;
            if (row < m && col < n) {{
                c[row * n + col] = sum[mi][ni];
            }}
        }}
    }}
}}

void {kernel.launcher_name}(
    {_SIGNATURE}) {{
    dim3 block({x}, {y});
    dim3 grid(cdiv(n, {bn}), cdiv(m, {bm}));
    {kernel.kernel_name}<<<grid, block>>>(a, b, c, m, n, k);
}}
"""


def cuda_source(kernel: ChecklistKernel) -> str:
    """Return the pinned text of a coarsened or a blocked kernel."""

    checked = require_kernel(kernel)
    if type(checked) is CoarsenedKernel:
        return coarsened_cuda_source(checked)
    return blocked_cuda_source(checked)


def source_files() -> tuple[tuple[str, str], ...]:
    """Return the file name and the text of each pinned source.

    The Day 76 cuBLAS call comes last, with its Day 76 text and digest, so a
    later run can time each kernel against cuBLAS from one set of files.
    """

    return (
        *((f"{kernel.name}.cu", cuda_source(kernel)) for kernel in PINNED_KERNELS),
        (CUBLAS_FILE, cublas_source()),
    )
