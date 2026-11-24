"""CPU references of flash attention: the blockwise float32 loop and float64 naive.

``blockwise_attention`` is the algorithm of the pinned Triton text in NumPy. It
walks the same query blocks and key blocks, with the same zero-padded tiles,
the same ``-inf`` masks and the same online softmax update, in float32. It
never forms an N x N matrix. ``naive_attention`` is the textbook form in
float64: it forms the full score matrix ``S = Q K^T * scale`` and the full
probability matrix ``P = softmax(S)``, and returns ``P V`` and the row
log-sum-exp. Both take the float32 scale of the kernel, so they define the
same function, and the float64 form is the contract of every run.

Each case draws its own float32 Q, K and V of shape ``[BH, N, D]`` from
``numpy.random.default_rng([seed, stream, index])``. Family ``normal`` holds
standard-normal values. Family ``large`` multiplies Q and K by 8, an exact
power of two, so the scores reach several hundred. Then ``exp(s)`` without the
max shift overflows float32 above 88.7, and the online softmax must still give
finite, accurate rows.

``attention_bounds`` declares the error bound of each head. It is a first-order
rounding analysis in unit roundoff ``u = 2**-24``, times a safety factor of 2.
It is not a proof:

* ``sigma = gamma_D * a_max + 2 u S`` bounds the error of one score, where
  ``a_max`` is the largest ``(|q| . |k|) * scale`` and S the largest ``|s|``;
* ``E = 2 sigma + 4 u S + 8 u (n_kb + 1)`` bounds the log error of one weight:
  the score and max errors, the subtractions, and an exp error of 8 u for each
  of the ``n_kb + 1`` exponentials that build a weight;
* ``R = 2 gamma_(N + n_kb) + 2 u (n_kb + 1)`` bounds the sums of ``l`` and
  ``acc`` and the rescale products;
* the output bound is ``2 ((exp(2 E) - 1) + R + u) * max|V|``, because each
  output row is a convex mix of value rows, and the log-sum-exp bound is
  ``2 (E + R)``.

The bound is for the NumPy exp of the CPU interpreter, and no interpreter
run checked it. A GPU ``tl.exp`` compiles to an approximate base-2
exponential, so a later GPU run needs its own bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy

from .kernel_metal_reference import MAX_SEED
from .matmul_metal_mapping import MatmulMetalError
from .matmul_metal_reference import FLOAT32_UNIT_ROUNDOFF, gamma, matrix_sha256
from .triton_flash_kernel import (
    HEAD_DIMS,
    TritonFlashError,
    _integer,
    _require_head_dim,
    float32_scale,
)

NORMAL = "normal"
LARGE = "large"
FAMILIES = (NORMAL, LARGE)
FAMILY_STREAMS = {NORMAL: 0, LARGE: 1}
LARGE_FACTOR = 8.0
MAX_CASE_INDEX = 63
MAX_BATCH_HEADS = 16
MAX_SEQUENCE = 1024
SAFETY = 2.0
EXP_ERROR_UNITS = 8
INPUTS = (
    "numpy.random.default_rng([seed, stream, index]) draws standard-normal "
    "float32 Q, then K, then V of shape [BH, N, D]; stream 0 is family normal "
    "and stream 1 is family large, where Q and K are multiplied by 8"
)
REFERENCE = (
    "float64 naive attention: S = Q K^T * scale with the float32 scale, the "
    "causal or key mask as -inf, P = softmax(S) with the row max shift, O = P V "
    "and LSE = max + log(sum exp(S - max))"
)
BOUND = (
    "per head: sigma = gamma_D * a_max + 2u * S; E = 2 sigma + 4u * S + "
    "8u * (n_kb + 1); R = 2 gamma_(N + n_kb) + 2u * (n_kb + 1); output bound "
    "2 * ((exp(2E) - 1) + R + u) * max|V|; LSE bound 2 * (E + R); u = 2**-24, "
    "gamma_n = n u / (1 - n u), a first-order analysis times 2, not a proof"
)


@dataclass(frozen=True)
class AttentionCase:
    """One attention shape: heads of all batches, sequence length, head dim."""

    batch_heads: int
    sequence: int
    head_dim: int
    causal: bool

    def __post_init__(self) -> None:
        _integer(self.batch_heads, "batch_heads", 1, MAX_BATCH_HEADS)
        _integer(self.sequence, "sequence", 1, MAX_SEQUENCE)
        _require_head_dim(self.head_dim)
        if type(self.causal) is not bool:
            raise TypeError("causal must be a bool")

    @property
    def scale(self) -> float:
        return float32_scale(self.head_dim)

    @property
    def elements(self) -> int:
        """Return the float32 values of one Q, K, V or O: ``BH * N * D``."""

        return self.batch_heads * self.sequence * self.head_dim

    def to_mapping(self) -> dict[str, object]:
        return {
            "batch_heads": self.batch_heads,
            "sequence": self.sequence,
            "head_dim": self.head_dim,
            "causal": self.causal,
        }


def require_case(case: object) -> AttentionCase:
    if type(case) is not AttentionCase:
        raise TypeError("case must be AttentionCase")
    return case


def attention_inputs(
    family: str, seed: int, index: int, case: AttentionCase
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Draw the float32 Q, K and V of one family of one case."""

    if family not in FAMILY_STREAMS:
        raise TritonFlashError(f"family must be one of {FAMILIES}")
    _integer(seed, "seed", 0, MAX_SEED)
    _integer(index, "case index", 0, MAX_CASE_INDEX)
    checked = require_case(case)
    shape = (checked.batch_heads, checked.sequence, checked.head_dim)
    generator = numpy.random.default_rng([seed, FAMILY_STREAMS[family], index])
    q, k, v = (generator.standard_normal(shape, dtype=numpy.float32) for _ in range(3))
    if family == LARGE:
        q = q * numpy.float32(LARGE_FACTOR)
        k = k * numpy.float32(LARGE_FACTOR)
    return q, k, v


def _operands(*arrays: object) -> tuple[numpy.ndarray, ...]:
    if (
        not all(
            isinstance(array, numpy.ndarray)
            and array.dtype == numpy.float32
            and array.ndim == 3
            for array in arrays
        )
        or len({array.shape for array in arrays}) != 1
    ):
        raise TritonFlashError("Q, K and V must be float32 arrays of one [BH, N, D]")
    if arrays[0].shape[2] not in HEAD_DIMS:
        raise TritonFlashError(f"head_dim must be one of {HEAD_DIMS}")
    return arrays


def visible(sequence: int, causal: bool) -> numpy.ndarray:
    """Return the N x N mask of the keys that each query row may see."""

    if not causal:
        return numpy.ones((sequence, sequence), dtype=bool)
    return numpy.tril(numpy.ones((sequence, sequence), dtype=bool))


def naive_attention(
    q: numpy.ndarray, k: numpy.ndarray, v: numpy.ndarray, causal: bool
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return float64 ``O`` and the row log-sum-exp; forms S and P in full."""

    q, k, v = _operands(q, k, v)
    scale = float32_scale(q.shape[2])
    q64, k64, v64 = (array.astype(numpy.float64) for array in (q, k, v))
    scores = numpy.matmul(q64, k64.transpose(0, 2, 1)) * scale
    scores[:, ~visible(q.shape[1], causal)] = -numpy.inf
    top = scores.max(axis=2, keepdims=True)
    weights = numpy.exp(scores - top)
    total = weights.sum(axis=2, keepdims=True)
    output = numpy.matmul(weights / total, v64)
    return output, (top + numpy.log(total))[:, :, 0]


def naive_float32(
    q: numpy.ndarray, k: numpy.ndarray, v: numpy.ndarray, causal: bool
) -> numpy.ndarray:
    """Return float32 attention that forms S and P, for a memory comparison."""

    q, k, v = _operands(q, k, v)
    scores = numpy.matmul(q, k.transpose(0, 2, 1)) * numpy.float32(
        float32_scale(q.shape[2])
    )
    scores[:, ~visible(q.shape[1], causal)] = -numpy.inf
    scores -= scores.max(axis=2, keepdims=True)
    numpy.exp(scores, out=scores)
    scores /= scores.sum(axis=2, keepdims=True)
    return numpy.matmul(scores, v)


def unshifted_overflows(q: numpy.ndarray, k: numpy.ndarray, causal: bool) -> int:
    """Count the visible float32 ``exp(s)`` that overflow without the max shift."""

    q, k = _operands(q, k)
    scores = numpy.matmul(q, k.transpose(0, 2, 1)) * numpy.float32(
        float32_scale(q.shape[2])
    )
    with numpy.errstate(over="ignore"):
        exponentials = numpy.exp(scores)
    return int(
        numpy.count_nonzero(numpy.isinf(exponentials[:, visible(q.shape[1], causal)]))
    )


def _tile(source: numpy.ndarray, start: int, rows: int) -> numpy.ndarray:
    """Return ``rows`` rows from ``start``, with 0 past the end: a masked load."""

    tile = numpy.zeros((rows, source.shape[1]), dtype=numpy.float32)
    block = source[start : start + rows]
    tile[: block.shape[0]] = block
    return tile


def blockwise_attention(
    q: numpy.ndarray,
    k: numpy.ndarray,
    v: numpy.ndarray,
    causal: bool,
    block_m: int,
    block_n: int,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Run the online softmax of the kernel in float32; return O, m and l.

    Each step mirrors one line of the text, in its order, on tiles of the
    kernel's shapes, so a CPU emulation of the text gives these bits.
    """

    q, k, v = _operands(q, k, v)
    if type(causal) is not bool:
        raise TypeError("causal must be a bool")
    heads, sequence, dim = q.shape
    scale = numpy.float32(float32_scale(dim))
    output = numpy.zeros(q.shape, dtype=numpy.float32)
    maxima = numpy.zeros((heads, sequence), dtype=numpy.float32)
    sums = numpy.zeros((heads, sequence), dtype=numpy.float32)
    zero = numpy.float32(0.0)
    for head in range(heads):
        for start_m in range(0, sequence, block_m):
            offs_m = start_m + numpy.arange(block_m)
            q_tile = _tile(q[head], start_m, block_m)
            m_i = numpy.zeros(block_m, dtype=numpy.float32) - numpy.float32(numpy.inf)
            l_i = numpy.zeros(block_m, dtype=numpy.float32)
            acc = numpy.zeros((block_m, dim), dtype=numpy.float32)
            for start_n in range(0, sequence, block_n):
                offs_n = start_n + numpy.arange(block_n)
                kt = _tile(k[head], start_n, block_n).T
                v_tile = _tile(v[head], start_n, block_n)
                s = (numpy.matmul(q_tile, kt) + zero) * scale
                keep = (offs_n < sequence)[None, :]
                if causal:
                    keep = keep & (offs_n[None, :] <= offs_m[:, None])
                s = numpy.where(keep, s, numpy.float32(-numpy.inf))
                m_new = numpy.maximum(m_i, s.max(axis=1))
                alpha = numpy.exp(m_i - m_new)
                p = numpy.exp(s - m_new[:, None])
                l_i = l_i * alpha + p.sum(axis=1)
                acc = acc * alpha[:, None]
                acc = numpy.matmul(p, v_tile) + acc
                m_i = m_new
            acc = acc / l_i[:, None]
            stop = min(block_m, sequence - start_m)
            output[head, start_m : start_m + stop] = acc[:stop]
            maxima[head, start_m : start_m + stop] = m_i[:stop]
            sums[head, start_m : start_m + stop] = l_i[:stop]
    return output, maxima, sums


@dataclass(frozen=True)
class HeadBound:
    """The declared output and log-sum-exp bounds of one head."""

    output: float
    lse: float


def attention_bounds(
    q: numpy.ndarray, k: numpy.ndarray, v: numpy.ndarray, causal: bool, block_n: int
) -> tuple[HeadBound, ...]:
    """Return the declared bound of each head for one key block size."""

    q, k, v = _operands(q, k, v)
    heads, sequence, dim = q.shape
    scale = float32_scale(dim)
    mask = visible(sequence, causal)
    u = FLOAT32_UNIT_ROUNDOFF
    blocks = -(-sequence // _integer(block_n, "block_n", 1, MAX_SEQUENCE))
    rest = 2 * gamma(sequence + blocks, u) + 2 * u * (blocks + 1)
    bounds = []
    for head in range(heads):
        q64, k64 = q[head].astype(numpy.float64), k[head].astype(numpy.float64)
        a_max = float((numpy.abs(q64) @ numpy.abs(k64).T * scale)[mask].max())
        s_max = float(numpy.abs(q64 @ k64.T * scale)[mask].max())
        sigma = gamma(dim, u) * a_max + 2 * u * s_max
        log_error = 2 * sigma + 4 * u * s_max + EXP_ERROR_UNITS * u * (blocks + 1)
        v_max = float(numpy.abs(v[head]).max())
        bounds.append(
            HeadBound(
                output=SAFETY * (math.expm1(2 * log_error) + rest + u) * v_max,
                lse=SAFETY * (log_error + rest),
            )
        )
    return tuple(bounds)


def array_sha256(array: numpy.ndarray) -> str:
    """Hash the C-order little-endian bytes of a float32 or float64 array."""

    try:
        return matrix_sha256(array)
    except MatmulMetalError as error:
        raise TritonFlashError(str(error)) from error
