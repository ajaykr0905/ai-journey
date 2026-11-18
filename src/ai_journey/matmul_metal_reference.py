"""Seeded matmul operands, a float64 CPU reference and a rounding bound in K.

Every case draws from its own stream, ``numpy.random.default_rng([seed,
purpose, index])``, so a run and an independent verifier rebuild identical
standard-normal float32 operands from the seed alone. Purpose 0 holds the
correctness cases and purpose 1 the benchmark points.

The reference adds the K products of each output in float64, in increasing k
order. A product of two float32 values is exact in float64, so the only
roundings are the K float64 additions, and the fixed order gives the same
bits on every IEEE host. ``numpy.matmul`` would not, because its order
depends on the BLAS build. The same loop adds the absolute products into
``|A| @ |B|``, the scale of the bound.

Bound: an inner product of length K that rounds each operation once in unit
roundoff u satisfies ``|fl(x . y) - x . y| <= gamma_K(u) |x| . |y|`` with
``gamma_K(u) = K u / (1 - K u)``, in any summation order and with or without
fused multiply-adds (N. J. Higham, Accuracy and Stability of Numerical
Algorithms, 2nd ed., SIAM, 2002, section 3.1). The float32 output and the
float64 reference each obey it, so every output element must satisfy
``|C - R| <= (gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)``. The
relative error of an element is ``|C - R| / (|A| @ |B|)``, and a case
records the largest one. The bound grows linearly in K; it is a worst case,
and random rounding errors usually stay far below it.

``fma_replay`` replays the kernels' arithmetic on the CPU bit for bit, so a
report can name the exact bits each Metal output must hold. Digests hash the
C-order little-endian bytes of an array, so equal digests mean bit-identical
arrays.
"""

from __future__ import annotations

from hashlib import sha256

import numpy

from .kernel_metal_geometry import require_integer
from .kernel_metal_reference import MAX_SEED
from .matmul_metal_mapping import MAX_DIMENSION, MatmulMetalError, matmul_shape
from .memory_access_tiling import MatmulShape

CORRECTNESS_STREAM = 0
BENCHMARK_STREAM = 1
MAX_CASE_INDEX = 63
FLOAT32_UNIT_ROUNDOFF = 2.0**-24
FLOAT64_UNIT_ROUNDOFF = 2.0**-53
ERROR_BOUND = "(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)"
GAMMA = "gamma_K(u) = K * u / (1 - K * u)"
RELATIVE_ERROR = "max over elements of |C - R| / (|A| @ |B|), in float64"
REFERENCE = (
    "float64 sum of the exact float32 products in increasing k order, on the "
    "CPU, rebuilt from the seed"
)
BOUND_SOURCE = (
    "N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., "
    "SIAM, 2002, section 3.1"
)
_LITTLE = {4: "<f4", 8: "<f8"}


def operand_generator(seed: int, stream: int, index: int) -> numpy.random.Generator:
    """Return the generator of one case, independent of every other case."""

    require_integer(seed, "seed", 0, MAX_SEED)
    require_integer(stream, "stream", CORRECTNESS_STREAM, BENCHMARK_STREAM)
    require_integer(index, "case index", 0, MAX_CASE_INDEX)
    return numpy.random.default_rng([seed, stream, index])


def matmul_inputs(
    seed: int, stream: int, index: int, shape: MatmulShape
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Draw standard-normal float32 ``A[m, k]`` and then ``B[k, n]``."""

    if not isinstance(shape, MatmulShape):
        raise TypeError("shape must be a MatmulShape")
    checked = matmul_shape(shape.m, shape.n, shape.k)
    generator = operand_generator(seed, stream, index)
    left = generator.standard_normal((checked.m, checked.k), dtype=numpy.float32)
    right = generator.standard_normal((checked.k, checked.n), dtype=numpy.float32)
    return left, right


def _operands(left: object, right: object) -> tuple[numpy.ndarray, numpy.ndarray]:
    if (
        not isinstance(left, numpy.ndarray)
        or not isinstance(right, numpy.ndarray)
        or left.dtype != numpy.float32
        or right.dtype != numpy.float32
        or left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[0]
    ):
        raise MatmulMetalError("operands must be conformable 2D float32 arrays")
    matmul_shape(int(left.shape[0]), int(right.shape[1]), int(left.shape[1]))
    return left, right


def float64_reference(
    left: numpy.ndarray, right: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return ``R = A @ B`` and ``|A| @ |B|``, summed in float64 in k order.

    Each step adds one exact product per output, so the result does not
    depend on the BLAS that the host has.
    """

    a, b = _operands(left, right)
    a64, b64 = a.astype(numpy.float64), b.astype(numpy.float64)
    shape = (a.shape[0], b.shape[1])
    reference = numpy.zeros(shape, dtype=numpy.float64)
    magnitude = numpy.zeros(shape, dtype=numpy.float64)
    product = numpy.empty(shape, dtype=numpy.float64)
    for step in range(a.shape[1]):
        numpy.multiply.outer(a64[:, step], b64[step, :], out=product)
        reference += product
        numpy.abs(product, out=product)
        magnitude += product
    return reference, magnitude


def _round_fused(product: numpy.ndarray, total: numpy.ndarray) -> numpy.ndarray:
    """Round each exact ``product + total`` once to float32, ties to even.

    ``product`` holds exact float64 products and ``total`` float32 values
    widened to float64. ``s = fl64(product + total)`` is the double nearest
    the exact sum ``x``, and TwoSum gives ``e = x - s`` exactly. Rounding is
    monotonic and every float32 midpoint is a double, so ``s`` and ``x`` round
    to the same float32 unless ``s`` is a midpoint itself. Then the sign of
    ``e`` picks the side of ``x``, and ``e == 0`` is a true tie, which the
    plain conversion already rounds to even.
    """

    near = product + total
    virtual = near - product
    error = (product - (near - virtual)) + (total - virtual)
    rounded = near.astype(numpy.float32)
    rounded64 = rounded.astype(numpy.float64)
    up = numpy.nextafter(rounded, numpy.float32(numpy.inf))
    down = numpy.nextafter(rounded, numpy.float32(-numpy.inf))
    offset = near - rounded64
    above = (offset > 0) & (2 * offset == up.astype(numpy.float64) - rounded64)
    below = (offset < 0) & (-2 * offset == rounded64 - down.astype(numpy.float64))
    raise_up = above & (error > 0)
    lower = below & (error < 0)
    rounded[raise_up] = up[raise_up]
    rounded[lower] = down[lower]
    return rounded


def fma_replay(left: numpy.ndarray, right: numpy.ndarray) -> numpy.ndarray:
    """Replay ``sum = fma(a[i, k], b[k, j], sum)`` for k in increasing order.

    This is the loop of both kernels: one rounding per step, starting from
    0. A product of two float32 values is exact in float64, so each step
    needs only the exact rounding of one sum. Standard-normal operands make
    a subnormal partial sum, where a GPU may flush to zero, vanishingly
    unlikely.
    """

    a, b = _operands(left, right)
    a64, b64 = a.astype(numpy.float64), b.astype(numpy.float64)
    shape = (a.shape[0], b.shape[1])
    total = numpy.zeros(shape, dtype=numpy.float32)
    product = numpy.empty(shape, dtype=numpy.float64)
    for step in range(a.shape[1]):
        numpy.multiply.outer(a64[:, step], b64[step, :], out=product)
        total = _round_fused(product, total.astype(numpy.float64))
    return total


def gamma(inner: int, unit_roundoff: float) -> float:
    """Return ``gamma_K(u) = K u / (1 - K u)``, the inner-product rounding factor."""

    product = require_integer(inner, "k", 1, MAX_DIMENSION) * unit_roundoff
    return product / (1 - product)


def error_bound(inner: int) -> float:
    """Return the largest relative error a float32 output may have for this K."""

    return gamma(inner, FLOAT32_UNIT_ROUNDOFF) + gamma(inner, FLOAT64_UNIT_ROUNDOFF)


def max_relative_error(
    output: numpy.ndarray, reference: numpy.ndarray, magnitude: numpy.ndarray
) -> float:
    """Return the largest ``|C - R| / (|A| @ |B|)`` over the output.

    An element with a zero scale counts as 0 when it matches the reference
    and as infinity when it does not, so no mismatch can hide behind a
    division by zero.
    """

    if (
        not isinstance(output, numpy.ndarray)
        or output.dtype != numpy.float32
        or not isinstance(reference, numpy.ndarray)
        or not isinstance(magnitude, numpy.ndarray)
        or output.shape != reference.shape
        or magnitude.shape != reference.shape
    ):
        raise MatmulMetalError("an output must be float32 in the reference shape")
    observed = output.astype(numpy.float64)
    if not numpy.isfinite(observed).all():
        raise MatmulMetalError("an output holds a NaN or an infinity")
    error = numpy.abs(observed - reference)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        numpy.divide(error, magnitude, out=error)
    # 0 / 0 is NaN, which fmax skips; a nonzero error over 0 is infinity.
    return float(numpy.fmax.reduce(error, axis=None, initial=0.0))


def matrix_sha256(array: numpy.ndarray) -> str:
    """Hash the C-order little-endian bytes of a float32 or float64 array."""

    if (
        not isinstance(array, numpy.ndarray)
        or array.dtype.kind != "f"
        or array.dtype.itemsize not in _LITTLE
    ):
        raise MatmulMetalError("only float32 and float64 arrays are hashed")
    little = numpy.ascontiguousarray(array, dtype=_LITTLE[array.dtype.itemsize])
    return sha256(little.data).hexdigest()
