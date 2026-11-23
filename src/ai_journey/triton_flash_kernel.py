"""Pinned Triton text of the Day 81 flash attention forward pass, and its configs.

``TRITON_SOURCE`` holds one ``@triton.jit`` kernel and one host launcher. The
launch grid is ``(cdiv(N, BLOCK_M), BH)``: one program for each query block of
each (batch, head) pair. Q, K, V and O are row-major float32 ``[BH, N, D]``
blocks. A program loads its ``BLOCK_M x D`` query tile once and then loops over
the key and value blocks of its head, ``BLOCK_N`` rows at a time. It never
writes a score or a probability to memory: the ``BLOCK_M x BLOCK_N`` score tile
lives only inside the loop.

The loop keeps the online softmax state of each query row: the running maximum
``m``, the running sum ``l`` of ``exp(s - m)`` and the unnormalized output
``acc``. When a key block raises the maximum, the program rescales ``l`` and
``acc`` by ``alpha = exp(m_old - m_new)``, so the old terms keep the new
reference point. After the last block, ``acc / l`` is the softmax-weighted sum
of the value rows. Each score is ``(q . k) * scale`` with the scale
``1 / sqrt(D)``, rounded to float32 by the host.

Masked loads give 0 for a row past N, and ``tl.where`` gives each score of a
key past N, or of a key after the query when ``CAUSAL`` is set, the value
``-inf``, so its weight is ``exp(-inf) = 0``. Thus a ragged N needs no padding.
The first key block holds key 0, which every query row may see, so ``m`` is
finite after the first block and no ``-inf - -inf`` occurs. The causal kernel
still visits the blocks above the diagonal and masks them. It is a simplified
kernel: FlashAttention-2 skips those blocks. Both ``tl.dot`` calls take
``input_precision="ieee"``, so a GPU stays off TF32, and ``acc`` is float32.

The kernel also stores ``m`` and ``l`` of each row. They are the O(N) state of
the algorithm, and ``m + log(l)`` is the log-sum-exp that a backward pass
needs. ``FLASH_CONFIGS`` declares block sizes, warps and stages for a later GPU
run. The lab records them and tunes nothing. This module never imports triton,
so the package imports on a host without it. Triton has no macOS wheel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy

from .gpu_tiled_cuda import GpuTiledError
from .gpu_tiled_cuda import source_sha256 as _text_sha256
from .kernel_metal_geometry import KernelMetalError, ceil_div, require_integer

KERNEL_NAME = "flash_attention_forward"
LAUNCHER_NAME = "launch_flash_attention"
LANGUAGE = "Triton (Python)"
INPUT_PRECISION = "ieee"
ACCUMULATOR = "float32"
FLOAT32_BYTES = 4
# tl.dot needs at least 16 along each axis, and tl.arange a power of two.
MIN_BLOCK = 16
MAX_BLOCK = 128
HEAD_DIMS = (16, 32, 64, 128)
MAX_STAGES = 8
WARP_COUNTS = (1, 2, 4, 8)
ONLINE_SOFTMAX = (
    "m_new = tl.maximum(m_i, tl.max(s, axis=1))",
    "alpha = tl.exp(m_i - m_new)",
    "p = tl.exp(s - m_new[:, None])",
    "l_i = l_i * alpha + tl.sum(p, axis=1)",
    "acc = acc * alpha[:, None]",
    'acc = tl.dot(p, v, acc, input_precision="ieee")',
    "m_i = m_new",
)
SCORE = 's = tl.dot(q, kt, input_precision="ieee") * scale'
LOAD_MASKS = (
    "q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)",
    "kt = tl.load(kt_ptrs, mask=col_mask[None, :], other=0.0)",
    "v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0)",
)
CAUSAL_MASK = "keep = col_mask[None, :] & (offs_n[None, :] <= offs_m[:, None])"
KEY_MASK = "keep = col_mask[None, :]"
STORES = (
    "tl.store(o_ptrs, acc, mask=row_mask[:, None])",
    "tl.store(m_ptr + pid_bh * stride_mh + offs_m, m_i, mask=row_mask)",
    "tl.store(l_ptr + pid_bh * stride_mh + offs_m, l_i, mask=row_mask)",
)


class TritonFlashError(ValueError):
    """Raised when a Triton flash attention input, run or report breaks its rules."""


def _integer(value: object, name: str, low: int, high: int) -> int:
    try:
        return require_integer(value, name, low, high)
    except KernelMetalError as error:
        raise TritonFlashError(str(error)) from error


def _kernel_lines() -> list[str]:
    body = "    "
    loop = "        "
    return [
        "@triton.jit",
        f"def {KERNEL_NAME}(",
        "    q_ptr, k_ptr, v_ptr, o_ptr, m_ptr, l_ptr, N, scale,",
        "    stride_xh, stride_xn, stride_oh, stride_on, stride_mh,",
        "    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,",
        "    CAUSAL: tl.constexpr,",
        "):",
        f"{body}# One program: one query block of one (batch, head) pair.",
        f"{body}pid_m = tl.program_id(0)",
        f"{body}pid_bh = tl.program_id(1)",
        f"{body}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
        f"{body}offs_d = tl.arange(0, HEAD_DIM)",
        f"{body}row_mask = offs_m < N",
        f"{body}head = pid_bh * stride_xh",
        f"{body}q_ptrs = q_ptr + head + offs_m[:, None] * stride_xn + "
        "offs_d[None, :]",
        f"{body}{LOAD_MASKS[0]}",
        f"{body}# Online softmax state: running max, running sum, output.",
        f'{body}m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")',
        f"{body}l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)",
        f"{body}acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)",
        f"{body}for start_n in range(0, N, BLOCK_N):",
        f"{loop}offs_n = start_n + tl.arange(0, BLOCK_N)",
        f"{loop}col_mask = offs_n < N",
        f"{loop}# K is loaded as a HEAD_DIM x BLOCK_N tile, so no transpose.",
        f"{loop}kt_ptrs = k_ptr + head + offs_n[None, :] * stride_xn + "
        "offs_d[:, None]",
        f"{loop}v_ptrs = v_ptr + head + offs_n[:, None] * stride_xn + "
        "offs_d[None, :]",
        *(loop + line for line in LOAD_MASKS[1:]),
        f"{loop}{SCORE}",
        f"{loop}if CAUSAL:",
        f"{loop}    {CAUSAL_MASK}",
        f"{loop}else:",
        f"{loop}    {KEY_MASK}",
        f'{loop}s = tl.where(keep, s, float("-inf"))',
        f"{loop}# Rescale the old state when this block raises the row max.",
        *(loop + line for line in ONLINE_SOFTMAX),
        f"{body}acc = acc / l_i[:, None]",
        f"{body}o_ptrs = o_ptr + pid_bh * stride_oh + offs_m[:, None] * stride_on + "
        "offs_d[None, :]",
        *(body + line for line in STORES),
    ]


def _launcher_lines() -> list[str]:
    return [
        f"def {LAUNCHER_NAME}(",
        "    q, k, v, o, m, l, bh, n, head_dim, scale, rows, *, causal, block_m,",
        "    block_n, num_warps, num_stages,",
        "):",
        "    grid = (triton.cdiv(n, block_m), bh)",
        f"    {KERNEL_NAME}[grid](",
        "        q, k, v, o, m, l, n, scale,",
        "        n * head_dim, head_dim, rows * head_dim, head_dim, rows,",
        "        HEAD_DIM=head_dim, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal,",
        "        num_warps=num_warps, num_stages=num_stages,",
        "    )",
    ]


TRITON_SOURCE = "\n".join(
    [
        "# Triton source generated by ai_journey.triton_flash_kernel.",
        "# The lab runs this text only in the Triton CPU interpreter, with",
        "# TRITON_INTERPRET=1. No GPU compiled or ran it. One program computes",
        "# the attention output of one BLOCK_M query block of one (batch, head)",
        "# pair. It loops over BLOCK_N key blocks with an online softmax, so no",
        "# N x N score matrix is written. Q, K, V and O are row-major float32.",
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
        return _text_sha256(text)
    except GpuTiledError as error:
        raise TritonFlashError(str(error)) from error


def float32_scale(head_dim: int) -> float:
    """Return ``1 / sqrt(D)`` rounded to float32, as the host passes it.

    A Python float argument becomes a float32 scalar in the kernel, so the
    host rounds it first and the CPU references use the same value.
    """

    _require_head_dim(head_dim)
    return float(numpy.float32(1.0 / math.sqrt(head_dim)))


def _require_head_dim(head_dim: object) -> int:
    if type(head_dim) is not int:
        raise TypeError("head_dim must be an integer")
    if head_dim not in HEAD_DIMS:
        raise TritonFlashError(f"head_dim must be one of {HEAD_DIMS}")
    return head_dim


@dataclass(frozen=True)
class FlashConfig:
    """One set of block sizes and compile options for the kernel."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        for name in ("block_m", "block_n"):
            value = _integer(getattr(self, name), name, MIN_BLOCK, MAX_BLOCK)
            if value & (value - 1):
                raise TritonFlashError(f"{name} must be a power of two")
        if type(self.num_warps) is not int:
            raise TypeError("num_warps must be an integer")
        if self.num_warps not in WARP_COUNTS:
            raise TritonFlashError(f"num_warps must be one of {WARP_COUNTS}")
        _integer(self.num_stages, "num_stages", 1, MAX_STAGES)

    @property
    def name(self) -> str:
        return f"m{self.block_m}_n{self.block_n}_w{self.num_warps}_s{self.num_stages}"

    def tile_bytes(self, head_dim: int) -> int:
        """Float32 bytes of one Q tile, one K tile, one V tile and one score tile.

        The compiler chooses what it keeps in registers or shared memory, so
        this is a size of the working set of one loop step, not a measured
        footprint.
        """

        dim = _require_head_dim(head_dim)
        return (
            self.block_m * dim + 2 * self.block_n * dim + self.block_m * self.block_n
        ) * FLOAT32_BYTES

    def meta_parameters(self) -> dict[str, int]:
        """Return the block sizes as ``triton.Config`` keyword arguments."""

        return {"BLOCK_M": self.block_m, "BLOCK_N": self.block_n}

    def launcher_arguments(self) -> dict[str, int]:
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            **self.launcher_arguments(),
            "tile_bytes": {str(dim): self.tile_bytes(dim) for dim in HEAD_DIMS},
        }


# Declared, not tuned. The first two are larger tiles for a later GPU run. The
# last two are small tiles that give the interpreter many ragged blocks.
FLASH_CONFIGS = (
    FlashConfig(128, 64, 8, 3),
    FlashConfig(64, 64, 4, 3),
    FlashConfig(64, 32, 4, 4),
    FlashConfig(16, 16, 1, 1),
)
CONFIGS_BY_NAME = {config.name: config for config in FLASH_CONFIGS}


def require_config(config: object) -> FlashConfig:
    if type(config) is not FlashConfig:
        raise TypeError("config must be FlashConfig")
    return config


def query_blocks(sequence: int, config: FlashConfig) -> int:
    """Return the query blocks of one head: the first axis of the grid."""

    return ceil_div(sequence, require_config(config).block_m)


def key_blocks(sequence: int, config: FlashConfig) -> int:
    """Return the iterations of the key loop of one program."""

    return ceil_div(sequence, require_config(config).block_n)


def padded_rows(sequence: int, config: FlashConfig) -> int:
    """Return N rounded up to whole query blocks: the rows a program covers."""

    return query_blocks(sequence, config) * require_config(config).block_m
