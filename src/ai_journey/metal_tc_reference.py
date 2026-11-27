"""Seeded operands in three input types, the float64 reference and the bounds.

Every case and point draws standard-normal float32 ``A[m, k]`` and then
``B[k, n]`` from its own stream with the Day 74 ``matmul_inputs``, so a run
and the verifier rebuild the same operands from the seed. M2 reads the
operands rounded to half and M3 reads them rounded to bfloat, each to the
nearest value with ties to even. S1, M1 and the float32 PyTorch call read
the float32 draws.

The reference is the float64 product of the float32 draws, the problem
that every precision tries to solve. It uses the Day 74
``float64_reference``: the K exact products of each output are added in
increasing k order, so the bits do not depend on the BLAS of the host. The
same loop gives the scale ``S = |A| @ |B|``, and two more k-order loops
give the sums ``R_i`` of each row of ``|A|`` and ``C_j`` of each column of
``|B|``.

The declared bound of an output follows from two facts. An input type of
unit roundoff ``u`` stores each operand with ``|x' - x| <= u |x| + e``,
where ``e`` is the smallest normal of the type, so the term covers a
subnormal operand and a device that flushes it to zero. A float32
accumulator that rounds each product and each sum at most once gives
``|fl(x' . y') - x' . y'| <= gamma_K |x'| . |y'|`` in any order, with or
without fused operations (N. J. Higham, Accuracy and Stability of Numerical
Algorithms, 2nd ed., SIAM, 2002, section 3.1). The matrix path of the GPU
names no order, so the bound assumes only that rule, and a run that breaks
the bound stops. The flush of a subnormal partial result adds at most
``2 (K + 1) 2**-126``.

A PyTorch call names no kernel, so no derived bound applies to it. Its
output must stay within a sanity bound of each type, which only catches a
wrong result.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy

from .kernel_metal_reference import FLOAT32_SENTINEL_BITS, float32_sentinels
from .matmul_metal_reference import (
    BENCHMARK_STREAM,
    CORRECTNESS_STREAM,
    MAX_CASE_INDEX,
    float64_reference,
    gamma,
    matmul_inputs,
)
from .memory_access_tiling import MatmulShape
from .metal_flash_reference import check_rows, float64_digest
from .metal_fused_reference import float32_sha256
from .metal_tc_source import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    MAX_DIMENSION,
    PRECISIONS,
    MetalTcError,
    require_size,
)

STREAMS = (CORRECTNESS_STREAM, BENCHMARK_STREAM)
TAIL = 64
GUARD_BITS = FLOAT32_SENTINEL_BITS
SENTINEL_BITS = FLOAT32_SENTINEL_BITS
ACCUMULATION_UNIT = 2.0**-24
REFERENCE_UNIT = 2.0**-53
FLUSH = 2.0**-126
INPUT_UNIT = {FLOAT32: 0.0, FLOAT16: 2.0**-11, BFLOAT16: 2.0**-8}
INPUT_FLOOR = {FLOAT32: 2.0**-126, FLOAT16: 2.0**-14, BFLOAT16: 2.0**-126}
VENDOR_UNIT = {FLOAT32: 2.0**-12, FLOAT16: 2.0**-6, BFLOAT16: 2.0**-4}
VENDOR_FLOOR = 2.0**-100
DISTRIBUTION = (
    "A and then B from a standard normal, drawn as float32 with the Day 74 "
    "matmul_inputs; half and bfloat operands are the float32 draws rounded to "
    "nearest, ties to even"
)
REFERENCE = (
    "float64 product of the float32 draws, with the exact products of each "
    "output added in increasing k order; S = |A| @ |B| by the same loop, and "
    "the row sums of |A| and the column sums of |B| in k order"
)
BOUND = (
    "|C - R| <= (gamma_K(2**-24) (1 + u)**2 + 2 u + u**2 + gamma_K(2**-53)) S + "
    "e (1 + u) (1 + gamma_K(2**-24)) (R_i + C_j) + K e**2 (1 + gamma_K(2**-24)) "
    "+ 2 (K + 1) 2**-126, with gamma_K(v) = K v / (1 - K v); u = 0 and e = "
    "2**-126 for float32 inputs, u = 2**-11 and e = 2**-14 for half, u = 2**-8 "
    "and e = 2**-126 for bfloat; it assumes that the float32 accumulators round "
    "each product and each sum at most once, in any order"
)
VENDOR_BOUND = (
    "|C - R| <= v S + 2**-100 for each output of torch.matmul, with v = 2**-12 "
    "for float32, 2**-6 for float16 and 2**-4 for bfloat16: a sanity bound only, "
    "because PyTorch names no kernel and no derived bound applies"
)
RELATIVE_ERROR = "max over the checked outputs of |C - R| / S, in float64"


def _float32(values: object, label: str) -> numpy.ndarray:
    if not isinstance(values, numpy.ndarray) or values.dtype != numpy.float32:
        raise MetalTcError(f"{label} must be a float32 array")
    return values


def bfloat16_round(values: numpy.ndarray) -> numpy.ndarray:
    """Round finite float32 values to bfloat, nearest with ties to even.

    The result is float32 again: each value keeps only its top 16 bits, so
    it converts to bfloat without a second rounding.
    """

    bits = _float32(values, "values").view(numpy.uint32).astype(numpy.uint64)
    if not numpy.all(numpy.isfinite(values)):
        raise MetalTcError("only finite values are rounded to bfloat here")
    bias = 0x7FFF + ((bits >> 16) & 1)
    kept = ((bits + bias) & 0xFFFF0000).astype(numpy.uint32)
    return kept.view(numpy.float32).reshape(values.shape)


def rounded(values: numpy.ndarray, precision: str) -> numpy.ndarray:
    """Return float32 values that the input type stores exactly."""

    _float32(values, "values")
    if precision == FLOAT32:
        return values
    if precision == FLOAT16:
        return values.astype(numpy.float16).astype(numpy.float32)
    if precision == BFLOAT16:
        return bfloat16_round(values)
    raise MetalTcError(f"{precision!r} is not an input type of this lab")


def input_sha256(values: numpy.ndarray, precision: str) -> str:
    """Hash the operand bits in the width of its type, little-endian."""

    stored = rounded(values, precision)
    if precision == FLOAT32:
        return float32_sha256(stored)
    if precision == FLOAT16:
        data = numpy.ascontiguousarray(stored.astype(numpy.float16)).astype("<f2")
    else:
        data = (stored.view(numpy.uint32) >> 16).astype("<u2")
    return sha256(numpy.ascontiguousarray(data).tobytes()).hexdigest()


def guarded(values: numpy.ndarray) -> numpy.ndarray:
    """Flatten an operand and append ``TAIL`` quiet NaNs, which no read may reach."""

    guard = numpy.full(TAIL, GUARD_BITS, dtype=numpy.uint32).view(numpy.float32)
    return numpy.concatenate((_float32(values, "operand").reshape(-1), guard))


def sentinel_output(shape: MatmulShape) -> numpy.ndarray:
    """Return a C buffer of ``m n + TAIL`` quiet-NaN sentinels with fixed bits."""

    return float32_sentinels(shape.m * shape.n + TAIL)


@dataclass(frozen=True)
class TcOperands:
    """The float32 draws of one case or point, with its shape."""

    shape: MatmulShape
    a: numpy.ndarray
    b: numpy.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.shape, MatmulShape):
            raise TypeError("shape must be a MatmulShape")
        m, n, k = self.shape.m, self.shape.n, self.shape.k
        for name, values, rows, columns in (
            ("a", self.a, m, k),
            ("b", self.b, k, n),
        ):
            if _float32(values, name).shape != (rows, columns):
                raise MetalTcError(f"{name} must be {rows} x {columns}")

    def inputs(self, precision: str) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Both operands as the input type stores them, in float32."""

        return rounded(self.a, precision), rounded(self.b, precision)

    def digests(self) -> dict[str, dict[str, str]]:
        """The digest of each operand in each input type."""

        return {
            precision: {
                "a_sha256": input_sha256(self.a, precision),
                "b_sha256": input_sha256(self.b, precision),
            }
            for precision in PRECISIONS
        }


def operands(seed: int, stream: int, index: int, shape: MatmulShape) -> TcOperands:
    """Draw the operands of one case or point from its own stream."""

    if stream not in STREAMS:
        raise MetalTcError(f"stream must be one of {STREAMS}")
    require_size(index, "case index", 0, MAX_CASE_INDEX)
    for name in ("m", "n", "k"):
        require_size(getattr(shape, name), name, 1, MAX_DIMENSION)
    left, right = matmul_inputs(seed, stream, index, shape)
    return TcOperands(shape, left, right)


def bound_terms(precision: str, k: int) -> tuple[float, float, float]:
    """Return the factors of ``S``, of ``R_i + C_j`` and the constant of the bound."""

    if precision not in PRECISIONS:
        raise MetalTcError(f"{precision!r} is not an input type of this lab")
    u, e = INPUT_UNIT[precision], INPUT_FLOOR[precision]
    g = gamma(k, ACCUMULATION_UNIT)
    r = gamma(k, REFERENCE_UNIT)
    scale = g * (1.0 + u) * (1.0 + u) + 2.0 * u + u * u + r
    sums = e * (1.0 + u) * (1.0 + g)
    constant = k * e * e * (1.0 + g) + 2.0 * (k + 1) * FLUSH
    return scale, sums, constant


def _k_sums(values: numpy.ndarray, axis: int) -> numpy.ndarray:
    """Add ``|values|`` along k one step at a time, so every host gets the bits."""

    magnitude = numpy.abs(values.astype(numpy.float64))
    if axis == 1:
        total = numpy.zeros(magnitude.shape[0], dtype=numpy.float64)
        for step in range(magnitude.shape[1]):
            total += magnitude[:, step]
        return total
    total = numpy.zeros(magnitude.shape[1], dtype=numpy.float64)
    for step in range(magnitude.shape[0]):
        total += magnitude[step, :]
    return total


@dataclass(frozen=True)
class TcReference:
    """The float64 product at the checked rows, with its scale and sums."""

    rows: numpy.ndarray
    output: numpy.ndarray
    magnitude: numpy.ndarray
    row_sums: numpy.ndarray
    column_sums: numpy.ndarray
    k: int

    def bound(self, precision: str) -> numpy.ndarray:
        """The declared bound of each checked output for one input type."""

        scale, sums, constant = bound_terms(precision, self.k)
        both = self.row_sums[:, None] + self.column_sums[None, :]
        return scale * self.magnitude + sums * both + constant

    def vendor_bound(self, precision: str) -> numpy.ndarray:
        """The sanity bound of each checked output of the PyTorch call."""

        if precision not in PRECISIONS:
            raise MetalTcError(f"{precision!r} is not an input type of this lab")
        return VENDOR_UNIT[precision] * self.magnitude + VENDOR_FLOOR

    def digests(self) -> dict[str, str]:
        """The digests of the reference and of its scale."""

        return {
            "reference_sha256": float64_digest(self.output),
            "magnitude_sha256": float64_digest(self.magnitude),
        }


def tc_reference(values: TcOperands, count: int | None = None) -> TcReference:
    """Compute the reference at every row, or at ``count`` rows spread evenly."""

    if not isinstance(values, TcOperands):
        raise TypeError("values must be TcOperands")
    rows = check_rows(values.shape.m, count)
    left = numpy.ascontiguousarray(values.a[rows])
    output, magnitude = float64_reference(left, values.b)
    if not numpy.all(magnitude > 0):
        raise MetalTcError("every checked output needs a positive scale")
    return TcReference(
        rows=rows,
        output=output,
        magnitude=magnitude,
        row_sums=_k_sums(left, 1),
        column_sums=_k_sums(values.b, 0),
        k=values.shape.k,
    )
