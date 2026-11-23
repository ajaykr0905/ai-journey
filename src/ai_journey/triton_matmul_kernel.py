"""Pinned Triton text of the Day 79 float32 matmul, with its declared configs.

``TRITON_SOURCE`` holds one ``@triton.jit`` kernel and one host launcher. One
program instance computes one ``BLOCK_M x BLOCK_N`` tile of ``C = A @ B``, all
three row-major float32. The launch grid has one axis of
``cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)`` programs, and ``GROUPED_ORDER`` maps
each program id to one tile of the 2D tile grid of C: ``GROUP_M`` rows of
tiles take consecutive ids, one tile column at a time. The Triton matmul
tutorial uses this order to reuse data in the L2 cache. This lab measures no
cache, so ``program_tile`` replays the order on the CPU only to show that it
covers each tile once.

Masks on both loads fill each slot outside A or B with 0, as the guarded loads
of the Day 76 CUDA text do, and a mask on the store keeps each write inside C,
so a ragged M, N or K needs no padding. The K loop adds the product of an A
tile and a B tile to a float32 accumulator with ``tl.dot``. Its argument
``input_precision="ieee"`` keeps a GPU off TF32 Tensor Core math, which is the
float32 default of ``tl.dot`` on GPUs with Tensor Cores, so the Day 74 float32
bound stays the contract of a later GPU run. The order of the additions inside
one ``tl.dot`` is not fixed: the compiler, or ``numpy.matmul`` in the CPU
interpreter, chooses it. Thus no CPU replay names the output bits, and the
correctness harness holds each output to a bound, not to bit equality.

Triton 3.9.0 removed block pointers: ``tl.make_block_ptr`` raises
``NotImplementedError``. Its tensor descriptors need row strides that are a
multiple of 16 bytes, and Day 74 shapes such as N = 33 do not have them. Thus
the kernel uses pointer arithmetic with masks.

``AUTOTUNE_CONFIGS`` declares the block sizes, warps and stages that a later
GPU run gives to ``triton.autotune`` with ``AUTOTUNE_KEY``. The lab records the
list and tunes nothing. This module never imports triton, so the package
imports on a host without it. Triton has no macOS wheel.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gpu_tiled_cuda import GpuTiledError
from .gpu_tiled_cuda import source_sha256 as _cuda_text_sha256
from .kernel_metal_geometry import KernelMetalError, ceil_div, require_integer
from .matmul_metal_mapping import MAX_DIMENSION
from .memory_access_tiling import MatmulShape

KERNEL_NAME = "matmul_kernel"
LAUNCHER_NAME = "launch_matmul"
LANGUAGE = "Triton (Python)"
INPUT_PRECISION = "ieee"
ACCUMULATOR = "float32"
AUTOTUNE_KEY = ("M", "N", "K")
FLOAT32_BYTES = 4
# tl.dot needs at least 16 along each block axis, and tl.arange a power of two.
MIN_BLOCK = 16
MAX_BLOCK = 256
MAX_GROUP_M = 64
MAX_STAGES = 8
WARP_COUNTS = (1, 2, 4, 8, 16)
TUTORIAL = (
    "python/tutorials/03-matrix-multiplication.py of triton-lang/triton at tag "
    "v3.9.0, get_cuda_autotune_config"
)
GROUPED_ORDER = (
    "pid = tl.program_id(axis=0)",
    "num_pid_m = tl.cdiv(M, BLOCK_M)",
    "num_pid_n = tl.cdiv(N, BLOCK_N)",
    "num_pid_in_group = GROUP_M * num_pid_n",
    "group_id = pid // num_pid_in_group",
    "first_pid_m = group_id * GROUP_M",
    "group_size_m = min(num_pid_m - first_pid_m, GROUP_M)",
    "pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)",
    "pid_n = (pid % num_pid_in_group) // group_size_m",
)
DOT = 'acc = tl.dot(a, b, acc, input_precision="ieee")'
LOAD_MASKS = (
    "a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k0)",
    "b_mask = (offs_k[:, None] < K - k0) & (offs_n[None, :] < N)",
)
STORE_MASK = "c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)"


class TritonMatmulError(ValueError):
    """Raised when a Triton matmul lab input, run or report breaks its contract."""


def _integer(value: object, name: str, low: int, high: int) -> int:
    try:
        return require_integer(value, name, low, high)
    except KernelMetalError as error:
        raise TritonMatmulError(str(error)) from error


def _kernel_lines() -> list[str]:
    body = "    "
    loop = "        "
    return [
        "@triton.jit",
        f"def {KERNEL_NAME}(",
        "    a_ptr, b_ptr, c_ptr, M, N, K,",
        "    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,",
        "    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,",
        "    GROUP_M: tl.constexpr,",
        "):",
        f"{body}# Grouped order: GROUP_M rows of tiles take consecutive program ids.",
        *(body + line for line in GROUPED_ORDER),
        f"{body}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
        f"{body}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
        f"{body}offs_k = tl.arange(0, BLOCK_K)",
        f"{body}a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * "
        "stride_ak",
        f"{body}b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * "
        "stride_bn",
        f"{body}acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)",
        f"{body}for k0 in range(0, K, BLOCK_K):",
        f"{loop}# A slot past M, past N or past the end of K loads 0.",
        *(loop + line for line in LOAD_MASKS),
        f"{loop}a = tl.load(a_ptrs, mask=a_mask, other=0.0)",
        f"{loop}b = tl.load(b_ptrs, mask=b_mask, other=0.0)",
        f"{loop}{DOT}",
        f"{loop}a_ptrs += BLOCK_K * stride_ak",
        f"{loop}b_ptrs += BLOCK_K * stride_bk",
        f"{body}c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * "
        "stride_cn",
        f"{body}{STORE_MASK}",
        f"{body}tl.store(c_ptrs, acc, mask=c_mask)",
    ]


def _launcher_lines() -> list[str]:
    return [
        f"def {LAUNCHER_NAME}(",
        "    a, b, c, m, n, k, *, block_m, block_n, block_k, group_m, num_warps, "
        "num_stages",
        "):",
        "    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)",
        f"    {KERNEL_NAME}[grid](",
        "        a, b, c, m, n, k, k, 1, n, 1, n, 1,",
        "        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,",
        "        num_warps=num_warps, num_stages=num_stages,",
        "    )",
    ]


TRITON_SOURCE = "\n".join(
    [
        "# Triton source generated by ai_journey.triton_matmul_kernel.",
        "# The lab ran this text only in the Triton CPU interpreter, with",
        "# TRITON_INTERPRET=1. No GPU compiled or ran it. It is source for a",
        "# later run on a GPU. One program computes one BLOCK_M x BLOCK_N tile of",
        "# C = A @ B, and all three matrices are row-major float32. Masks fill",
        "# each slot outside A or B with 0 and keep each store inside C. tl.dot",
        "# adds each tile product to a float32 accumulator in IEEE float32",
        "# arithmetic, with no TF32.",
        "import triton",
        "import triton.language as tl",
        "",
        "",
        *_kernel_lines(),
        "",
        "",
        *_launcher_lines(),
        "",
    ]
)


def source_sha256(text: str = TRITON_SOURCE) -> str:
    """Return the SHA-256 of an ASCII text, by default of ``TRITON_SOURCE``."""

    try:
        return _cuda_text_sha256(text)
    except GpuTiledError as error:
        raise TritonMatmulError(str(error)) from error


def require_shape(shape: object) -> MatmulShape:
    """Return a shape whose int32 offsets stay exact in the kernel.

    With at most 8192 per dimension, the largest offset that the kernel forms,
    ``offs_m * stride_am + offs_k``, stays below 2**27, inside int32.
    """

    if type(shape) is not MatmulShape:
        raise TypeError("shape must be MatmulShape")
    if max(shape.m, shape.n, shape.k) > MAX_DIMENSION:
        raise TritonMatmulError(f"each dimension must be at most {MAX_DIMENSION}")
    return shape


@dataclass(frozen=True)
class TritonConfig:
    """One set of meta-parameters and compile options for ``matmul_kernel``."""

    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        for name in ("block_m", "block_n", "block_k"):
            value = _integer(getattr(self, name), name, MIN_BLOCK, MAX_BLOCK)
            if value & (value - 1):
                raise TritonMatmulError(f"{name} must be a power of two")
        _integer(self.group_m, "group_m", 1, MAX_GROUP_M)
        if type(self.num_warps) is not int:
            raise TypeError("num_warps must be an integer")
        if self.num_warps not in WARP_COUNTS:
            raise TritonMatmulError(f"num_warps must be one of {WARP_COUNTS}")
        _integer(self.num_stages, "num_stages", 1, MAX_STAGES)

    @property
    def name(self) -> str:
        return (
            f"m{self.block_m}_n{self.block_n}_k{self.block_k}_g{self.group_m}"
            f"_w{self.num_warps}_s{self.num_stages}"
        )

    @property
    def tile_bytes(self) -> int:
        """Float32 bytes of one A tile and one B tile.

        The compiler decides how many copies of them it keeps in shared
        memory for ``num_stages``, so this is not the shared memory of a
        launch.
        """

        return (
            self.block_m * self.block_k + self.block_k * self.block_n
        ) * FLOAT32_BYTES

    def meta_parameters(self) -> dict[str, int]:
        """Return the ``tl.constexpr`` arguments, as ``triton.Config`` kwargs."""

        return {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k,
            "GROUP_M": self.group_m,
        }

    def launcher_arguments(self) -> dict[str, int]:
        """Return the keyword arguments of ``launch_matmul`` in the text."""

        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "group_m": self.group_m,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            **self.launcher_arguments(),
            "tile_bytes": self.tile_bytes,
        }


# The six configs of the tutorial whose tiles have at most 128 rows and 128
# columns, with K blocks of 32. Its other CUDA configs target float16 or fp8,
# and a float32 tile has twice the bytes of a float16 tile.
AUTOTUNE_CONFIGS = (
    TritonConfig(128, 128, 32, 8, 4, 4),
    TritonConfig(128, 64, 32, 8, 4, 4),
    TritonConfig(64, 128, 32, 8, 4, 4),
    TritonConfig(128, 32, 32, 8, 4, 4),
    TritonConfig(64, 32, 32, 8, 2, 5),
    TritonConfig(32, 64, 32, 8, 2, 5),
)
CONFIGS_BY_NAME = {config.name: config for config in AUTOTUNE_CONFIGS}


def require_config(config: object) -> TritonConfig:
    if type(config) is not TritonConfig:
        raise TypeError("config must be TritonConfig")
    return config


def tile_grid(shape: MatmulShape, config: TritonConfig) -> tuple[int, int]:
    """Return ``(num_pid_m, num_pid_n)``: the rows and columns of C tiles."""

    checked, chosen = require_shape(shape), require_config(config)
    return ceil_div(checked.m, chosen.block_m), ceil_div(checked.n, chosen.block_n)


def programs(shape: MatmulShape, config: TritonConfig) -> int:
    """Return the programs of the one-axis launch grid: one per tile of C."""

    rows, columns = tile_grid(shape, config)
    return rows * columns


def k_blocks(shape: MatmulShape, config: TritonConfig) -> int:
    """Return the iterations of the K loop: ``cdiv(K, BLOCK_K)``."""

    return ceil_div(require_shape(shape).k, require_config(config).block_k)


def program_tile(pid: int, rows: int, columns: int, group_m: int) -> tuple[int, int]:
    """Replay ``GROUPED_ORDER`` for one program id: return ``(pid_m, pid_n)``."""

    _integer(rows, "num_pid_m", 1, MAX_DIMENSION)
    _integer(columns, "num_pid_n", 1, MAX_DIMENSION)
    _integer(group_m, "group_m", 1, MAX_GROUP_M)
    _integer(pid, "pid", 0, rows * columns - 1)
    in_group = group_m * columns
    first = pid // in_group * group_m
    size = min(rows - first, group_m)
    return first + pid % in_group % size, pid % in_group // size
