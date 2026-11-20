"""Seeded scan inputs, the float64 reference, CPU replays and per-element bounds.

Every case draws from its own stream, ``numpy.random.default_rng([seed,
stream, index])``, so a run and an independent verifier rebuild the same
input from the seed alone. Two patterns cover two questions:

- ``standard_normal``: float32 normal draws. An output must stay within the
  bound of its kernel's order, element by element. The text of S0 and S1
  fixes every addition, so the CPU replays them in float32, and a run
  records whether the GPU held the replay's bits. That is a measurement.
- ``small_integers``: whole numbers from -4 to 4, at most ``MAX_EXACT_LENGTH``
  of them. Every partial sum is an integer of at most 2^24 in magnitude, and
  float32 holds each one exactly, so any order of additions gives the exact
  prefix sums. Every output must equal them bit for bit.

The bound of element ``i`` is ``c * sum(|x[j]|, j <= i)`` with ``c =
(gamma_D(2**-24) + gamma_N(2**-53)) / (1 - gamma_N(2**-53))`` and
``gamma_k(u) = k u / (1 - k u)`` (Higham, section 4.2). ``D`` is the declared
rounding depth of the kernel at that length, the most float32 additions on a
path from an input to an output. The float64 reference is ``numpy.cumsum``
in order, so its own error is at most ``gamma_N(2**-53)`` times the same sum,
and the last factor covers the float64 rounding of that sum. A SIMD-group
prefix sum adds in an order that Metal does not state, so S2 declares the
worst order: the bound assumes only that each value enters once.
"""

from __future__ import annotations

import math
from hashlib import sha256

import numpy

from .gpu_scan_source import (
    BLOCK,
    KERNELS,
    MAX_LENGTH,
    S0_HILLIS_STEELE,
    S1_BLELLOCH,
    GpuScanError,
    block_count,
    require_count,
    require_length,
    rounding_depth,
)
from .kernel_metal_reference import MAX_SEED

CORRECTNESS_STREAM = 0
BENCHMARK_STREAM = 1
MAX_CASE_INDEX = 63
NORMAL = "standard_normal"
INTEGERS = "small_integers"
PATTERNS = (NORMAL, INTEGERS)
INTEGER_LIMIT = 4
MAX_EXACT_LENGTH = 1 << 22
U32 = 2.0**-24
U64 = 2.0**-53
ERROR_RULE = (
    "max over i of |output[i] - reference[i]| / sum(|x[j]|, j <= i), with the "
    "float64 reference and sum from numpy.cumsum; an equal element counts 0"
)
BOUND_RULE = (
    "(gamma_D(2**-24) + gamma_N(2**-53)) / (1 - gamma_N(2**-53)), gamma_k(u) = "
    "k u / (1 - k u); D = levels * block_depth + levels - 1 of the kernel at "
    "length N"
)
BOUND_SOURCE = (
    "N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM, "
    "2002, section 4.2: a sum whose terms pass at most D additions has error at "
    "most gamma_D times the sum of their absolute values"
)


def scan_inputs(
    seed: int, stream: int, index: int, length: int, pattern: str
) -> numpy.ndarray:
    """Draw the float32 input of one case or sweep point from its own stream."""

    require_count(seed, "seed", 0, MAX_SEED)
    require_count(stream, "stream", CORRECTNESS_STREAM, BENCHMARK_STREAM)
    require_count(index, "case index", 0, MAX_CASE_INDEX)
    count = require_length(length)
    generator = numpy.random.default_rng([seed, stream, index])
    if pattern == NORMAL:
        return generator.standard_normal(count, dtype=numpy.float32)
    if pattern == INTEGERS:
        if count > MAX_EXACT_LENGTH:
            raise GpuScanError(
                f"an integer case holds at most {MAX_EXACT_LENGTH} elements"
            )
        drawn = generator.integers(-INTEGER_LIMIT, INTEGER_LIMIT + 1, size=count)
        return drawn.astype(numpy.float32)
    raise GpuScanError(f"pattern must be one of {', '.join(PATTERNS)}")


def _values(values: object) -> numpy.ndarray:
    if (
        not isinstance(values, numpy.ndarray)
        or values.dtype != numpy.float32
        or values.ndim != 1
        or not 1 <= values.size <= MAX_LENGTH
    ):
        raise GpuScanError(f"a scan input is a float32 vector of 1 to {MAX_LENGTH}")
    return numpy.ascontiguousarray(values)


def float64_reference(values: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Sum the inputs in float64 in order, with the running sum of their sizes."""

    wide = _values(values).astype(numpy.float64)
    return numpy.cumsum(wide), numpy.cumsum(numpy.abs(wide))


def float64_sha256(array: numpy.ndarray) -> str:
    """Hash the little-endian C-order bytes of a float64 vector."""

    if (
        not isinstance(array, numpy.ndarray)
        or array.dtype.kind != "f"
        or array.dtype.itemsize != 8
    ):
        raise GpuScanError("only float64 arrays are hashed here")
    little = array.astype(numpy.dtype("<f8"), copy=False)
    return sha256(numpy.ascontiguousarray(little).data).hexdigest()


def exact_prefix(values: numpy.ndarray) -> numpy.ndarray:
    """The exact prefix sums of a ``small_integers`` input, as float32."""

    data = _values(values)
    whole = data.astype(numpy.int64)
    if (
        data.size > MAX_EXACT_LENGTH
        or not numpy.array_equal(whole.astype(numpy.float32), data)
        or int(numpy.abs(whole).max()) > INTEGER_LIMIT
    ):
        raise GpuScanError("exact prefix sums need small integers")
    return numpy.cumsum(whole).astype(numpy.float32)


def _blocks(values: numpy.ndarray) -> numpy.ndarray:
    """Zero-pad to whole blocks, one block for each row, as the kernels load it."""

    data = numpy.zeros(block_count(values.size) * BLOCK, dtype=numpy.float32)
    data[: values.size] = values
    return data.reshape(-1, BLOCK)


def _hillis_steele_blocks(values: numpy.ndarray) -> tuple[numpy.ndarray, ...]:
    data = _blocks(values)
    offset = 1
    while offset < BLOCK:
        later = data.copy()
        numpy.add(data[:, :-offset], data[:, offset:], out=later[:, offset:])
        data, offset = later, offset * 2
    return data.reshape(-1)[: values.size].copy(), data[:, -1].copy()


def _blelloch_blocks(values: numpy.ndarray) -> tuple[numpy.ndarray, ...]:
    tree = _blocks(values)
    stride = 1
    while stride < BLOCK:
        tree[:, 2 * stride - 1 :: 2 * stride] += tree[:, stride - 1 :: 2 * stride]
        stride *= 2
    total = tree[:, -1].copy()
    tree[:, -1] = 0.0
    stride = BLOCK // 2
    while stride:
        right = tree[:, 2 * stride - 1 :: 2 * stride]
        left = tree[:, stride - 1 :: 2 * stride]
        carry = left.copy()
        left[...] = right
        right += carry
        stride //= 2
    inclusive = numpy.empty_like(tree)
    inclusive[:, :-1] = tree[:, 1:]
    inclusive[:, -1] = total
    return inclusive.reshape(-1)[: values.size].copy(), total


def _levels(values: numpy.ndarray, block_scan) -> numpy.ndarray:
    """Scan the block sums with the same order, then add them, as the GPU does."""

    prefix, totals = block_scan(values)
    if totals.size == 1:
        return prefix
    offsets = _levels(totals, block_scan)
    prefix[BLOCK:] += numpy.repeat(offsets[:-1], BLOCK)[: values.size - BLOCK]
    return prefix


def hillis_steele_replay(values: numpy.ndarray) -> numpy.ndarray:
    """Replay S0's float32 additions in its order over every level."""

    return _levels(_values(values), _hillis_steele_blocks)


def blelloch_replay(values: numpy.ndarray) -> numpy.ndarray:
    """Replay S1's float32 additions in its order over every level."""

    return _levels(_values(values), _blelloch_blocks)


REPLAYS = {
    S0_HILLIS_STEELE: hillis_steele_replay,
    S1_BLELLOCH: blelloch_replay,
}
if set(REPLAYS) != {name for name, kernel in KERNELS.items() if kernel.replay}:
    raise GpuScanError("every kernel that declares a replay needs one")


def gamma(count: int, unit: float) -> float:
    """Return ``gamma_k(u) = k u / (1 - k u)``, defined while ``k u < 1``."""

    product = count * unit
    if not 0.0 <= product < 1.0:
        raise GpuScanError("gamma needs k u below 1")
    return product / (1.0 - product)


def error_bound(name: str, length: int) -> float:
    """The relative bound of one kernel's output at one length, by ``BOUND_RULE``."""

    depth = rounding_depth(name, length)
    wide = gamma(length, U64)
    return (gamma(depth, U32) + wide) / (1.0 - wide)


def max_relative_error(
    output: numpy.ndarray, reference: numpy.ndarray, scale: numpy.ndarray
) -> float:
    """Return the largest error over scale, or infinity for a value that is not finite.

    An element equal to the reference counts 0, even where the scale is 0.
    """

    if (
        not isinstance(output, numpy.ndarray)
        or output.dtype != numpy.float32
        or output.shape != reference.shape
    ):
        raise GpuScanError("an output must be a float32 vector of the input length")
    if not numpy.isfinite(output).all():
        return math.inf
    difference = numpy.abs(output.astype(numpy.float64) - reference)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        ratio = numpy.where(difference == 0.0, 0.0, difference / scale)
    return float(ratio.max())
