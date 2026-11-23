"""Seeded inputs, exact sums, launches and the float32 error bound of each order.

Every correctness case draws from its own stream,
``numpy.random.default_rng([seed, REDUCE_STREAM, index])``, so a run and an
independent verifier rebuild identical float32 inputs from the seed alone.
The patterns are ``normal`` (standard normal), ``positive`` (uniform in
[0, 1)), ``mixed`` (a sign times [1, 2) times 2^e for e from -24 to 24),
``ones`` and ``halves`` (k / 2 for k from -8 to 8). The timing input is one
standard-normal array of the largest size, from ``BENCHMARK_STREAM``; each
size reads its first n values.

Each float32 value is an integer times 2^-149, so the exact sum is an
integer too. ``exact_sums`` adds the integer significands of each binary
exponent in float64, where every partial sum stays an integer below 2^53
and is thus exact, then joins the exponents with Python integers. The
reference is that exact sum rounded once to float64.

The bound follows the summation order of each kernel. Any order of
additions is a binary tree, and a value passes through at most D roundings
on its way to the root. With round to nearest, the error is at most
``gamma_D(u) * sum |x|``, where ``u = 2^-24`` and ``gamma_D = D u / (1 - D u)``
(N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.,
SIAM, 2002, section 4.2). D adds four parts:

- the serial chain of one thread: 0 for one element, 1 for two, and
  ``ceil(n / grid) - 1`` for the R5 loop. Any order of c values in one
  thread has a depth of at most c - 1, so a compiler that reorders the
  loop keeps the bound.
- the tree of one group: 8 levels for the threadgroup trees of R0 to R3.
  A ``simd_sum`` does not state its order, so it counts as the worst tree:
  31 for 32 lanes, and 7 for the 8 group sums, because an addition of an
  exact zero rounds nothing.
- the combine chain: ``ceil(G / 1024) - 1`` for G partials.
- the combine tree: 10 levels for 1024 threads.

The model needs normal results. Each input value is a multiple of 2^g for
the smallest such g over the values, and so is every partial sum. A lab
input therefore needs ``g >= -126``: then no nonzero partial sum is
subnormal, and a flush to zero changes no result. When ``sum |x| <= 2^24 *
2^g``, every partial sum of every order is exact in float32, and the
result must equal the exact sum.

PyTorch does not state the order of ``torch.sum``. Its result gets the
declared tolerance ``gamma_4096(u) * sum |x|``, not a bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy

from .gpu_reduce_plan import MAX_ELEMENTS
from .gpu_reduce_source import (
    COMBINE,
    COMBINE_GROUP_SIZE,
    GRID_STRIDE,
    INTERLEAVED_TREE,
    KERNELS,
    ONE_PER_THREAD,
    SEQUENTIAL_TREE,
    SIMD_GROUPS,
    SIMD_SUM,
    SIMD_WIDTH,
    STEP_KERNELS,
    TREE_THEN_SHUFFLE,
    TWO_PER_THREAD,
    GpuReduceError,
    ReduceKernel,
    require_count,
)
from .kernel_metal_geometry import LaunchGeometry, ceil_div, vector_launch
from .kernel_metal_reference import MAX_SEED

REDUCE_STREAM = 78
BENCHMARK_STREAM = 79
MAX_CASE_INDEX = 63
NORMAL = "normal"
POSITIVE = "positive"
MIXED = "mixed"
ONES = "ones"
HALVES = "halves"
PATTERNS = (NORMAL, POSITIVE, MIXED, ONES, HALVES)
MIXED_EXPONENTS = (-24, 24)
HALF_STEPS = (-8, 8)
UNIT_ROUNDOFF = Fraction(1, 1 << 24)
SIGNIFICAND_BITS = 24
# Every finite float32 is an integer times 2^-149.
SCALE_EXPONENT = -149
MIN_GRID_EXPONENT = -126
VENDOR_DEPTH = 4096
CHUNK = 1 << 22
MAX_DEPTH = 1 << 20
BOUND_SOURCE = (
    "N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., "
    "SIAM, 2002, section 4.2: an order whose values pass through at most D "
    "roundings errs by at most gamma_D(u) * sum |x|, u = 2^-24"
)
DEPTH_RULE = (
    "D = thread chain + group tree + combine chain + combine tree: chain 0 for "
    "one element, 1 for two, ceil(n / 262144) - 1 for the R5 loop; tree 8 for "
    "the threadgroup trees of R0 to R3, 31 + 7 for the two simd_sum steps of "
    "R4 and R5; combine chain ceil(G / 1024) - 1 for G partials; combine tree "
    "10"
)
VENDOR_RULE = (
    "torch.sum states no order, so its result gets the declared tolerance "
    "gamma_4096(u) * sum |x|; it is a tolerance, not a derived bound"
)
EXACT_RULE = (
    "when sum |x| <= 2^24 * 2^g, with 2^g the finest power of two that divides "
    "every value, every partial sum of every order is exact in float32, so the "
    "result must equal the exact sum"
)


def case_generator(seed: int, stream: int, index: int) -> numpy.random.Generator:
    """Return the generator of one case, independent of every other case."""

    require_count(seed, "seed", 0, MAX_SEED)
    if stream not in (REDUCE_STREAM, BENCHMARK_STREAM):
        raise GpuReduceError("stream must be the reduce or the benchmark stream")
    require_count(index, "case index", 0, MAX_CASE_INDEX)
    return numpy.random.default_rng([seed, stream, index])


def case_values(seed: int, index: int, pattern: str, n: int) -> numpy.ndarray:
    """Draw the float32 input of one correctness case."""

    count = require_count(n, "n", 1, MAX_ELEMENTS)
    generator = case_generator(seed, REDUCE_STREAM, index)
    if pattern == NORMAL:
        return generator.standard_normal(count, dtype=numpy.float32)
    if pattern == POSITIVE:
        return generator.random(count, dtype=numpy.float32)
    if pattern == MIXED:
        low, high = MIXED_EXPONENTS
        signs = generator.choice(numpy.array([-1.0, 1.0]), size=count)
        mantissas = 1.0 + generator.random(count, dtype=numpy.float32)
        exponents = generator.integers(low, high + 1, size=count)
        values = signs * numpy.ldexp(mantissas.astype(numpy.float64), exponents)
        return values.astype(numpy.float32)
    if pattern == ONES:
        return numpy.ones(count, dtype=numpy.float32)
    if pattern == HALVES:
        low, high = HALF_STEPS
        steps = generator.integers(low, high + 1, size=count)
        return (steps * 0.5).astype(numpy.float32)
    raise GpuReduceError(f"pattern must be one of {', '.join(PATTERNS)}")


def benchmark_values(seed: int, n: int) -> numpy.ndarray:
    """Draw the one standard-normal timing input; each size reads a prefix."""

    count = require_count(n, "n", 1, MAX_ELEMENTS)
    generator = case_generator(seed, BENCHMARK_STREAM, 0)
    return generator.standard_normal(count, dtype=numpy.float32)


def _require_values(values: object) -> numpy.ndarray:
    if (
        not isinstance(values, numpy.ndarray)
        or values.dtype != numpy.float32
        or values.ndim != 1
        or not 1 <= values.size <= MAX_ELEMENTS
    ):
        raise GpuReduceError("values must be a 1D float32 array of 1 to 2^26 values")
    if not numpy.isfinite(values).all():
        raise GpuReduceError("values must be finite")
    return numpy.ascontiguousarray(values)


def _fields(chunk: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Split float32 bits into sign, integer significand and binary shift."""

    bits = chunk.view(numpy.uint32)
    exponent = ((bits >> 23) & 0xFF).astype(numpy.int64)
    significand = (bits & 0x7FFFFF).astype(numpy.int64)
    normal = exponent > 0
    significand[normal] |= 1 << 23
    shift = numpy.where(normal, exponent - 1, 0)
    return (bits >> 31).astype(bool), significand, shift


def _bucket_total(significands: numpy.ndarray, shifts: numpy.ndarray) -> int:
    """Join exact per-shift integer sums: value = significand * 2^(shift - 149)."""

    # A chunk sum stays below 2^22 * 2^24 = 2^46, so float64 adds it exactly.
    buckets = numpy.bincount(
        shifts, weights=significands.astype(numpy.float64), minlength=254
    )
    return sum(int(value) << shift for shift, value in enumerate(buckets) if value)


@dataclass(frozen=True)
class ExactSums:
    """The exact sum, the exact sum of magnitudes and the finest power of two."""

    total: Fraction
    magnitude: Fraction
    grid_exponent: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.total, Fraction) or not isinstance(
            self.magnitude, Fraction
        ):
            raise TypeError("exact sums are Fractions")
        if self.magnitude < abs(self.total):
            raise GpuReduceError("sum |x| cannot be below |sum x|")
        if (self.grid_exponent is None) != (self.magnitude == 0):
            raise GpuReduceError("only an all-zero input has no grid")

    @property
    def reference(self) -> float:
        """The exact sum rounded once to float64: the float64 reference."""

        return float(self.total)

    @property
    def sum_abs(self) -> float:
        """The exact sum of magnitudes rounded once to float64."""

        return float(self.magnitude)

    @property
    def exact_in_any_order(self) -> bool:
        """Whether every partial sum of every order is exact in float32."""

        if self.grid_exponent is None:
            return True
        return self.magnitude <= Fraction(2) ** (SIGNIFICAND_BITS + self.grid_exponent)

    def to_mapping(self) -> dict[str, object]:
        """Return the reference facts as a report records them."""

        return {
            "reference": self.reference,
            "sum_abs": self.sum_abs,
            "grid_exponent": self.grid_exponent,
            "exact_in_any_order": self.exact_in_any_order,
        }


class _Accumulator:
    """Exact running sums over chunks, so a 2^26-value input needs little memory."""

    def __init__(self) -> None:
        self.total = 0
        self.magnitude = 0
        self.grid: int | None = None

    def add(self, chunk: numpy.ndarray) -> None:
        negative, significands, shifts = _fields(chunk)
        self.magnitude += _bucket_total(significands, shifts)
        signed = numpy.where(negative, -significands, significands)
        self.total += _bucket_total(signed, shifts)
        nonzero = significands != 0
        if nonzero.any():
            low = significands[nonzero] & -significands[nonzero]
            # A power of two below 2^24 is exact in float64; frexp gives log2 + 1.
            _, powers = numpy.frexp(low.astype(numpy.float64))
            grid = int((shifts[nonzero] + powers - 1).min()) + SCALE_EXPONENT
            self.grid = grid if self.grid is None else min(self.grid, grid)

    def result(self) -> ExactSums:
        if self.grid is not None and self.grid < MIN_GRID_EXPONENT:
            raise GpuReduceError(
                "an input value is finer than 2^-126, so a partial sum could be "
                "subnormal and the bound would not hold"
            )
        scale = Fraction(1, 1 << -SCALE_EXPONENT)
        return ExactSums(self.total * scale, self.magnitude * scale, self.grid)


def exact_sums(values: numpy.ndarray) -> ExactSums:
    """Sum float32 values exactly, with their magnitudes and their finest grid."""

    array = _require_values(values)
    accumulator = _Accumulator()
    for start in range(0, array.size, CHUNK):
        accumulator.add(array[start : start + CHUNK])
    return accumulator.result()


def prefix_sums(values: numpy.ndarray, sizes: tuple[int, ...]) -> dict[int, ExactSums]:
    """Return the exact sums of each prefix ``values[:n]`` in one pass."""

    array = _require_values(values)
    wanted = sorted(set(sizes))
    if not wanted or wanted[0] < 1 or wanted[-1] > array.size:
        raise GpuReduceError("each prefix size must be 1 to the length of the values")
    accumulator, start, sums = _Accumulator(), 0, {}
    for size in wanted:
        for first in range(start, size, CHUNK):
            accumulator.add(array[first : min(first + CHUNK, size)])
        start = size
        sums[size] = accumulator.result()
    return sums


def _kernel(name: str) -> ReduceKernel:
    if name not in STEP_KERNELS:
        raise GpuReduceError(f"{name!r} is not a step kernel")
    return KERNELS[name]


def pass_one_launch(name: str, n: int) -> LaunchGeometry:
    """Size the pass-1 launch of a step kernel for n elements, in whole groups."""

    kernel = _kernel(name)
    count = require_count(n, "n", 1, MAX_ELEMENTS)
    if kernel.mapping == ONE_PER_THREAD:
        extent = count
    elif kernel.mapping == TWO_PER_THREAD:
        extent = ceil_div(count, 2)
    else:
        extent = kernel.fixed_groups * kernel.group_size
    return vector_launch(extent, kernel.group_size)


def combine_launch() -> LaunchGeometry:
    """The pass-2 launch: one group of 1024 threads for any partial count."""

    return vector_launch(COMBINE_GROUP_SIZE, KERNELS[COMBINE].group_size)


def partial_count(name: str, n: int) -> int:
    """Pass-1 groups for n elements: one partial sum for each group."""

    return pass_one_launch(name, n).groups[0]


def _tree_depth(kernel: ReduceKernel) -> int:
    levels = int(math.log2(kernel.group_size))
    if kernel.tree in (INTERLEAVED_TREE, SEQUENTIAL_TREE, TREE_THEN_SHUFFLE):
        return levels
    if kernel.tree == SIMD_SUM:
        return (SIMD_WIDTH - 1) + (SIMD_GROUPS - 1)
    raise GpuReduceError(f"{kernel.name} has no known tree")


def thread_chain_depth(name: str, n: int) -> int:
    """Roundings in one thread's own sum before the group adds it."""

    kernel = _kernel(name)
    count = require_count(n, "n", 1, MAX_ELEMENTS)
    if kernel.mapping == ONE_PER_THREAD:
        return 0
    if kernel.mapping == TWO_PER_THREAD:
        return 1
    if kernel.mapping == GRID_STRIDE:
        return ceil_div(count, kernel.fixed_groups * kernel.group_size) - 1
    raise GpuReduceError(f"{name} has no known mapping")


def combine_depth(partials: int) -> int:
    """Roundings of one partial in pass 2: its thread chain and the 10 levels."""

    count = require_count(partials, "partials", 1, MAX_ELEMENTS)
    levels = int(math.log2(COMBINE_GROUP_SIZE))
    return ceil_div(count, COMBINE_GROUP_SIZE) - 1 + levels


def summation_depth(name: str, n: int) -> int:
    """The most roundings that any element passes through, both passes included."""

    return (
        thread_chain_depth(name, n)
        + _tree_depth(_kernel(name))
        + combine_depth(partial_count(name, n))
    )


def gamma(depth: int) -> Fraction:
    """``gamma_D(u) = D u / (1 - D u)`` for ``u = 2^-24``, exactly."""

    count = require_count(depth, "depth", 0, MAX_DEPTH)
    product = count * UNIT_ROUNDOFF
    return product / (1 - product)


@dataclass(frozen=True)
class ReductionCheck:
    """One result against the exact sum: its error, bound and ratio, exactly."""

    output: float
    depth: int
    sums: ExactSums

    def __post_init__(self) -> None:
        if type(self.output) is not float or not math.isfinite(self.output):
            raise GpuReduceError("a reduction result must be a finite float")
        if float(numpy.float32(self.output)) != self.output:
            raise GpuReduceError("a reduction result must be a float32 value")
        require_count(self.depth, "depth", 0, MAX_DEPTH)
        if not isinstance(self.sums, ExactSums):
            raise TypeError("sums must be ExactSums")

    @property
    def error(self) -> Fraction:
        """``|output - exact sum|``, exactly."""

        return abs(Fraction(self.output) - self.sums.total)

    @property
    def bound(self) -> Fraction:
        """``gamma_D(u) * sum |x|``, exactly."""

        return gamma(self.depth) * self.sums.magnitude

    @property
    def within(self) -> bool:
        """The error is at most the bound, and zero when the sum is exact."""

        if self.sums.exact_in_any_order and self.error != 0:
            return False
        return self.error <= self.bound

    @property
    def bound_ratio(self) -> float:
        """Error over bound, rounded once; 0.0 when both are zero."""

        if self.bound == 0:
            return 0.0 if self.error == 0 else math.inf
        return float(self.error / self.bound)

    def to_mapping(self) -> dict[str, object]:
        """Return the check as a report records it."""

        return {
            "output": self.output,
            "depth": self.depth,
            "error": float(self.error),
            "bound": float(self.bound),
            "bound_ratio": self.bound_ratio,
            "exact_result": self.error == 0,
        }
