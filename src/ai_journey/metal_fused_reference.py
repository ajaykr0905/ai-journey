"""Seeded softmax inputs, row layouts and the float64 reference, with error bounds.

A layout lists the length of each row. Row ``r`` starts at ``starts[r]``,
and ``gap`` slots follow every row, so a correctness buffer holds a guard
of ``gap`` slots between rows and after the last one. Every gap slot of the
input holds ``GAP_INPUT``, so a kernel that reads one gets a wrong max, and
every gap slot of the output starts as a sentinel that a correct kernel
keeps. A timing layout packs its rows with no gap. A dense layout has rows
of one length; a ragged one has rows of many lengths.

Every case draws from its own stream, ``numpy.random.default_rng([seed,
stream, index])``, so a run and the verifier rebuild identical inputs from
the seed. The named distributions are part of the plan:

- ``normal``: a standard normal.
- ``uniform_80``: uniform from -80 to 80, so the ``exp`` arguments reach
  -160 and many results flush to zero.
- ``offset_plus_1000`` and ``offset_minus_1000``: a standard normal plus or
  minus 1000. Without the max subtraction, ``exp`` overflows or every term
  is 0, and the float32 softmax gives NaN; ``naive_float32_finite`` records
  that fact for each case.

The reference is the float64 softmax with the max subtraction, from the
float32 inputs. Its exp is ``exp64``, not ``numpy.exp``: the libm of a
platform rounds some float64 results to the other neighbor, and one changed
bit changes a digest. ``exp64`` uses only IEEE additions, products and
``ldexp``, so a report from one host verifies with the same bits on another
host. ``error_bounds`` gives a bound for each element and each step's order
of operations. With the unit roundoff ``u = 2**-24``, the distance
``d = max - x`` and ``EXP_ERROR`` for one call of ``precise::exp``:

- the computed ``exp(fl(x - max))`` has a relative error of at most
  ``alpha - 1``, with ``alpha = exp(d u) (1 + EXP_ERROR)``;
- each term of the sum passes ``A`` roundings and ``N`` calls of exp, so the
  sum has a relative error of at most ``rho = sum_j y_j (beta_j - 1)`` plus
  a flush term, with ``beta = exp(d u) (1 + EXP_ERROR)**N (1 + gamma_A)``;
- one correctly rounded division then gives the relative error
  ``alpha (1 + u) / (1 - rho) - 1``.

``A`` and ``N`` depend on the step, the row length and the group size. A
result below ``2**-126`` can flush to zero, so each bound adds ``FLOOR``.
The bound assumes that each ``precise::exp`` result has a relative error of
at most ``EXP_ULPS`` units of ``2**-23``; the run measures that assumption
with the exp probe and stops when it fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256

import numpy

from .kernel_metal_reference import FLOAT32_SENTINEL_BITS, MAX_SEED
from .metal_fused_source import (
    F1,
    F2,
    F3,
    LOWEST,
    MAX_ROW_LENGTH,
    SIMD_WIDTH,
    STEP_VARIANTS,
    U0,
    MetalFusedError,
    group_size,
    require_count,
    slots_used,
)

CORRECTNESS_STREAM = 0
BENCHMARK_STREAM = 1
RAGGED_STREAM = 2
STREAMS = (CORRECTNESS_STREAM, BENCHMARK_STREAM, RAGGED_STREAM)
MAX_CASE_INDEX = 63
MAX_ROWS = 1 << 16
MAX_ELEMENTS = 1 << 24
MAX_GAP = 1024
CORRECTNESS_GAP = 32
GAP_INPUT = 2.0**100
SENTINEL_BITS = FLOAT32_SENTINEL_BITS
NORMAL = "normal"
UNIFORM_80 = "uniform_80"
OFFSET_PLUS_1000 = "offset_plus_1000"
OFFSET_MINUS_1000 = "offset_minus_1000"
DISTRIBUTIONS = {
    NORMAL: "a standard normal, drawn as float32",
    UNIFORM_80: "uniform from -80 to 80, drawn in float64 and rounded to float32",
    OFFSET_PLUS_1000: "1000 plus a standard normal, rounded to float32",
    OFFSET_MINUS_1000: "-1000 plus a standard normal, rounded to float32",
}
UNIT = 2.0**-24
EXP_ULPS = 4
EXP_ERROR = EXP_ULPS * 2.0**-23
FLUSH = 2.0**-126
FLOOR = 2.0**-125
REFERENCE_UNIT = 2.0**-53
REFERENCE = (
    "float64 softmax of the float32 inputs, row by row: exp(x - max) divided by "
    "the sum of exp(x - max) over the row, with numpy.add.reduceat and exp64, "
    "an exp of IEEE additions, products and ldexp only, so each host computes "
    "the same bits"
)
BOUND = (
    "|y - y_ref| <= (rel + ref) / (1 - ref) * y_ref + 2**-125 for each element, "
    "with rel = alpha (1 + u) / (1 - rho) - 1, alpha = exp(d u) (1 + EXP_ERROR), "
    "rho = (sum_j y_ref_j (beta_j - 1)) / (1 - ref) + (n + 2 G) 2**-126, beta = "
    "exp(d u) (1 + EXP_ERROR)**N (1 + gamma_A), d = max - x, u = 2**-24, "
    "EXP_ERROR = 4 * 2**-23, gamma_A = A u / (1 - A u) and ref = gamma_(n+8) "
    "for the unit 2**-53"
)
ORDER_TERMS = {
    U0: "A = (k - 1) + log2(G) and N = 1: k slots added in order, then a tree",
    F1: "A = (k - 1) + log2(G) and N = 1: the additions of U0 in its order",
    F2: (
        "A = (k - 1) + 31 + (G / 32 - 1) and N = 1: k slots in order, simd_sum "
        "over 32 lanes, then simd_sum over the G / 32 partials"
    ),
    F3: (
        "A = (k - 1) + 1 + 31 + 1 + (G / 32 - 1) and N = k + 2: k online updates, "
        "a rescale and simd_sum in each SIMD group, a rescale and simd_sum of the "
        "partials; each update or rescale rounds once"
    ),
}
ROW_SUM_BOUND = (
    "|sum(y) - 1| <= the sum of the element bounds of the row + ref + 2 gamma_n "
    "for the unit 2**-53, where sum adds the outputs of the row in float64 in "
    "order"
)


# Cody and Waite's split of ln 2, as in fdlibm: LN2_HI has 32 significant
# bits, so k * LN2_HI and x - k * LN2_HI are exact for the clamped k.
_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_INV_LN2 = 1.44269504088896338700e00
# exp rounds to 0 below the low end, and the lab never needs the high end.
_EXP_LOW = -1100.0
_EXP_HIGH = 709.0
# 1/2!, 1/3!, ..., 1/14!: the Taylor terms of expm1(r) - r for |r| <= ln(2)/2.
_TAYLOR = tuple(1.0 / math.factorial(order) for order in range(2, 15))


def exp64(values: object) -> numpy.ndarray:
    """Return exp of finite float64 values, within 1 ulp, the same on every host.

    ``numpy.exp`` calls the libm of the platform, and two hosts can round
    one float64 result to different neighbors. Here every step is an IEEE
    addition, product or ``ldexp``, each correctly rounded, so every IEEE
    host gives the same bits. ``x = k ln 2 + r`` with ``|r| <= ln(2) / 2``;
    the Taylor terms give ``expm1(r)``, and the rounding error of ``1 + r``
    is kept exactly, so one rounding sets most of the error.
    """

    x = numpy.asarray(values, dtype=numpy.float64)
    if not numpy.all(numpy.isfinite(x)):
        raise MetalFusedError("exp64 takes finite values only")
    x = numpy.clip(x, _EXP_LOW, _EXP_HIGH)
    k = numpy.rint(x * _INV_LN2)
    high = x - k * _LN2_HI
    low = k * _LN2_LO
    r = high - low
    lost = (high - r) - low
    series = numpy.full_like(r, _TAYLOR[-1])
    for term in reversed(_TAYLOR[:-1]):
        series = series * r + term
    head = 1.0 + r
    tail = ((1.0 - head) + r) + (r * r * series + lost * (1.0 + r))
    return numpy.ldexp(head + tail, k.astype(numpy.int32))


def exp_growth(count: int) -> float:
    """Return ``(1 + EXP_ERROR) ** count`` by products, the same on every host."""

    value = 1.0
    for _ in range(count):
        value *= 1 + EXP_ERROR
    return value


def _gamma(count: int, unit: float) -> float:
    product = count * unit
    return product / (1 - product)


@dataclass(frozen=True)
class RowLayout:
    """The row lengths of one batch and the guard slots after each row."""

    lengths: tuple[int, ...]
    gap: int

    def __post_init__(self) -> None:
        if type(self.lengths) is not tuple or not 1 <= len(self.lengths) <= MAX_ROWS:
            raise MetalFusedError(f"a layout holds 1 to {MAX_ROWS} rows")
        for length in self.lengths:
            require_count(length, "row length", 1, MAX_ROW_LENGTH)
        require_count(self.gap, "gap", 0, MAX_GAP)
        if self.elements > MAX_ELEMENTS:
            raise MetalFusedError(
                f"a layout of {self.elements} elements exceeds {MAX_ELEMENTS}"
            )

    @property
    def rows(self) -> int:
        """The number of rows, one threadgroup each."""

        return len(self.lengths)

    @property
    def elements(self) -> int:
        """The softmax elements of every row, gaps excluded."""

        return sum(self.lengths)

    @property
    def longest(self) -> int:
        """The longest row, which sets the group size."""

        return max(self.lengths)

    @property
    def group_size(self) -> int:
        """The launch rule's G for this batch."""

        return group_size(self.longest)

    @property
    def buffer_elements(self) -> int:
        """The slots of an input or output buffer, gaps included."""

        return self.elements + self.gap * self.rows

    @property
    def dense(self) -> bool:
        """Whether every row has one length."""

        return len(set(self.lengths)) == 1

    def starts(self) -> numpy.ndarray:
        """The first slot of each row, as the int32 array the kernels read."""

        steps = numpy.asarray(self.lengths, dtype=numpy.int64) + self.gap
        starts = numpy.concatenate(([0], numpy.cumsum(steps)[:-1]))
        return starts.astype(numpy.int32)

    def lengths_array(self) -> numpy.ndarray:
        """The row lengths, as the int32 array the kernels read."""

        return numpy.asarray(self.lengths, dtype=numpy.int32)

    def valid(self) -> numpy.ndarray:
        """The buffer slots that hold a softmax element, in row order."""

        if self.gap == 0:
            return numpy.arange(self.elements, dtype=numpy.int64)
        starts = self.starts().astype(numpy.int64)
        lengths = self.lengths_array().astype(numpy.int64)
        packed = numpy.arange(self.elements, dtype=numpy.int64)
        packed_starts = numpy.concatenate(([0], numpy.cumsum(lengths)[:-1]))
        return packed + numpy.repeat(starts - packed_starts, lengths)

    def to_mapping(self) -> dict[str, object]:
        """Return the layout as a report records it, the lengths by digest."""

        return {
            "rows": self.rows,
            "elements": self.elements,
            "shortest": min(self.lengths),
            "longest": self.longest,
            "group_size": self.group_size,
            "gap": self.gap,
            "buffer_elements": self.buffer_elements,
            "dense": self.dense,
            "lengths_sha256": sha256(
                self.lengths_array().astype("<i4").tobytes()
            ).hexdigest(),
        }


def dense_layout(rows: int, cols: int, gap: int = 0) -> RowLayout:
    """A batch of ``rows`` rows of ``cols`` elements."""

    require_count(rows, "rows", 1, MAX_ROWS)
    return RowLayout((cols,) * rows, gap)


def _generator(seed: int, stream: int, index: int) -> numpy.random.Generator:
    require_count(seed, "seed", 0, MAX_SEED)
    if stream not in STREAMS:
        raise MetalFusedError(f"stream must be one of {STREAMS}")
    require_count(index, "case index", 0, MAX_CASE_INDEX)
    return numpy.random.default_rng([seed, stream, index])


def ragged_layout(seed: int, index: int, rows: int, max_length: int) -> RowLayout:
    """Draw ``rows`` lengths uniformly from 1 to ``max_length`` for one point."""

    require_count(rows, "rows", 1, MAX_ROWS)
    require_count(max_length, "max_length", 1, MAX_ROW_LENGTH)
    generator = _generator(seed, RAGGED_STREAM, index)
    lengths = generator.integers(1, max_length + 1, size=rows, dtype=numpy.int64)
    return RowLayout(tuple(int(length) for length in lengths), 0)


def draw_values(generator: numpy.random.Generator, count: int, distribution: str):
    """Draw ``count`` float32 values of one named distribution."""

    if distribution == NORMAL:
        return generator.standard_normal(count, dtype=numpy.float32)
    if distribution == UNIFORM_80:
        return generator.uniform(-80.0, 80.0, count).astype(numpy.float32)
    if distribution in (OFFSET_PLUS_1000, OFFSET_MINUS_1000):
        offset = 1000.0 if distribution == OFFSET_PLUS_1000 else -1000.0
        return (offset + generator.standard_normal(count)).astype(numpy.float32)
    raise MetalFusedError(f"distribution must be one of {', '.join(DISTRIBUTIONS)}")


def softmax_inputs(
    seed: int, stream: int, index: int, layout: RowLayout, distribution: str
) -> numpy.ndarray:
    """Return one input buffer: seeded rows, and ``GAP_INPUT`` in every gap."""

    if not isinstance(layout, RowLayout):
        raise TypeError("layout must be a RowLayout")
    values = draw_values(_generator(seed, stream, index), layout.elements, distribution)
    if layout.gap == 0:
        return values
    buffer = numpy.full(layout.buffer_elements, GAP_INPUT, dtype=numpy.float32)
    buffer[layout.valid()] = values
    return buffer


def sentinel_buffer(layout: RowLayout) -> numpy.ndarray:
    """Return an output buffer of quiet-NaN sentinels with fixed bits."""

    return numpy.full(layout.buffer_elements, SENTINEL_BITS, dtype=numpy.uint32).view(
        numpy.float32
    )


@dataclass(frozen=True)
class SoftmaxReference:
    """The float64 softmax of one buffer's rows, with each element's distance."""

    layout: RowLayout
    output: numpy.ndarray
    distance: numpy.ndarray

    def row_starts(self) -> numpy.ndarray:
        """The first packed element of each row."""

        lengths = self.layout.lengths_array().astype(numpy.int64)
        return numpy.concatenate(([0], numpy.cumsum(lengths)[:-1]))


def softmax_reference(buffer: numpy.ndarray, layout: RowLayout) -> SoftmaxReference:
    """Compute the float64 softmax of every row of one input buffer."""

    if (
        not isinstance(buffer, numpy.ndarray)
        or buffer.dtype != numpy.float32
        or buffer.shape != (layout.buffer_elements,)
    ):
        raise MetalFusedError("an input buffer must be float32 of the layout's size")
    values = buffer[layout.valid()].astype(numpy.float64)
    if not numpy.all(numpy.isfinite(values)):
        raise MetalFusedError("every softmax input must be finite")
    lengths = layout.lengths_array().astype(numpy.int64)
    starts = numpy.concatenate(([0], numpy.cumsum(lengths)[:-1]))
    row_max = numpy.maximum.reduceat(values, starts)
    distance = numpy.repeat(row_max, lengths) - values
    terms = exp64(-distance)
    totals = numpy.add.reduceat(terms, starts)
    return SoftmaxReference(layout, terms / numpy.repeat(totals, lengths), distance)


def order_terms(variant: str, length: int, size: int) -> tuple[int, int]:
    """Return ``(A, N)``: roundings and exp calls of one term in one step's order.

    ``k`` is the count of slots that thread 0 fills. A term passes at most
    ``k - 1`` thread additions, then the step's reduction; F3 also rescales
    the term at each later max of its thread and once in each merge.
    """

    used = slots_used(length, size)
    if variant in (U0, F1):
        return (used - 1) + size.bit_length() - 1, 1
    partials = size // SIMD_WIDTH - 1
    if variant == F2:
        return (used - 1) + (SIMD_WIDTH - 1) + partials, 1
    if variant == F3:
        return (used - 1) + 1 + (SIMD_WIDTH - 1) + 1 + partials, used + 2
    raise MetalFusedError(f"{variant!r} is not a registered step")


@dataclass(frozen=True)
class ErrorBounds:
    """The bound of each element and of each row sum, for one step's order."""

    elements: numpy.ndarray
    row_sums: numpy.ndarray


def error_bounds(variant: str, reference: SoftmaxReference) -> ErrorBounds:
    """Bound every element of one step's output, and each row sum, by ``BOUND``."""

    if variant not in STEP_VARIANTS:
        raise MetalFusedError(f"{variant!r} is not a registered step")
    layout = reference.layout
    size = layout.group_size
    lengths = layout.lengths_array().astype(numpy.int64)
    terms = [order_terms(variant, int(length), size) for length in layout.lengths]
    additions = numpy.repeat([float(a) for a, _ in terms], lengths)
    growth = numpy.repeat([exp_growth(n) for _, n in terms], lengths)
    ref = numpy.array([_gamma(int(n) + 8, REFERENCE_UNIT) for n in lengths])
    ref_each = numpy.repeat(ref, lengths)
    shift = exp64(reference.distance * UNIT)
    alpha = shift * (1 + EXP_ERROR)
    gamma = additions * UNIT / (1 - additions * UNIT)
    beta = shift * growth * (1 + gamma)
    starts = reference.row_starts()
    weighted = numpy.add.reduceat(reference.output * (beta - 1), starts)
    rho = weighted / (1 - ref) + (lengths + 2 * size) * FLUSH
    if not numpy.all(rho < 0.5):
        raise MetalFusedError("the sum bound is too weak to use")
    rel = alpha * (1 + UNIT) / (1 - numpy.repeat(rho, lengths)) - 1
    elements = (rel + ref_each) / (1 - ref_each) * reference.output + FLOOR
    summation = numpy.array([_gamma(int(n), REFERENCE_UNIT) for n in lengths])
    row_sums = numpy.add.reduceat(elements, starts) + ref + 2 * summation
    return ErrorBounds(elements, row_sums)


def naive_float32_finite(buffer: numpy.ndarray, layout: RowLayout) -> bool:
    """Whether exp(x) / sum(exp(x)) in float32, with no max subtraction, is finite.

    For the offset distributions it is not, which shows that a finite output
    needs the max subtraction.
    """

    values = buffer[layout.valid()]
    lengths = layout.lengths_array().astype(numpy.int64)
    starts = numpy.concatenate(([0], numpy.cumsum(lengths)[:-1]))
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        terms = numpy.exp(values)
        totals = numpy.add.reduceat(terms, starts, dtype=numpy.float32)
        output = terms / numpy.repeat(totals, lengths)
    return bool(numpy.all(numpy.isfinite(output)))


EXP_PROBE_COUNT = 1 << 18
EXP_PROBE_EXTREMES = (LOWEST, -1.0e30, -1000.0, -200.0, -150.0, -105.0, -87.5, -0.0)
EXP_PROBE = (
    "precise::exp at 2**18 float32 arguments: 262136 evenly spaced from -104 to "
    "0, then the extremes " + ", ".join(repr(value) for value in EXP_PROBE_EXTREMES)
)
EXP_PROBE_RULE = (
    "each result whose float64 exp is at least 2**-126 has a relative error of at "
    "most EXP_ERROR = 4 * 2**-23; each smaller result is within 2**-126 of it"
)


def exp_probe_arguments() -> numpy.ndarray:
    """Return the declared probe arguments, one float32 for each probe thread."""

    count = EXP_PROBE_COUNT - len(EXP_PROBE_EXTREMES)
    grid = numpy.linspace(-104.0, 0.0, count).astype(numpy.float32)
    extremes = numpy.asarray(EXP_PROBE_EXTREMES, dtype=numpy.float32)
    return numpy.concatenate((grid, extremes))


@dataclass(frozen=True)
class ExpProbe:
    """The measured error of one set of ``precise::exp`` results."""

    normal_results: int
    max_relative_error: float
    max_ulps: float
    tiny_results: int
    tiny_nonzero: int
    tiny_max_abs_error: float

    @property
    def holds(self) -> bool:
        """Whether the results meet ``EXP_PROBE_RULE``."""

        return self.max_relative_error <= EXP_ERROR and self.tiny_max_abs_error <= (
            FLUSH
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the measured errors as a report records them."""

        return {
            "normal_results": self.normal_results,
            "max_relative_error": self.max_relative_error,
            "max_ulps": self.max_ulps,
            "tiny_results": self.tiny_results,
            "tiny_nonzero": self.tiny_nonzero,
            "tiny_max_abs_error": self.tiny_max_abs_error,
            "holds": self.holds,
        }


def exp_probe_errors(results: numpy.ndarray) -> ExpProbe:
    """Measure the probe results against float64 exp of the probe arguments."""

    arguments = exp_probe_arguments()
    if (
        not isinstance(results, numpy.ndarray)
        or results.dtype != numpy.float32
        or results.shape != arguments.shape
    ):
        raise MetalFusedError("the exp probe must return one float32 per argument")
    exact = exp64(arguments.astype(numpy.float64))
    got = results.astype(numpy.float64)
    if not numpy.all(numpy.isfinite(got)):
        raise MetalFusedError("the exp probe returned a value that is not finite")
    normal = exact >= FLUSH
    error = numpy.abs(got - exact)
    relative = error[normal] / exact[normal]
    spacing = numpy.spacing(exact[normal].astype(numpy.float32)).astype(numpy.float64)
    tiny = ~normal
    return ExpProbe(
        normal_results=int(numpy.count_nonzero(normal)),
        max_relative_error=float(relative.max(initial=0.0)),
        max_ulps=float((error[normal] / spacing).max(initial=0.0)),
        tiny_results=int(numpy.count_nonzero(tiny)),
        tiny_nonzero=int(numpy.count_nonzero(got[tiny])),
        tiny_max_abs_error=float(error[tiny].max(initial=0.0)),
    )


def float32_sha256(array: numpy.ndarray) -> str:
    """Hash the little-endian bytes of a float32 array."""

    if not isinstance(array, numpy.ndarray) or array.dtype != numpy.float32:
        raise MetalFusedError("only float32 arrays are hashed here")
    return sha256(numpy.ascontiguousarray(array).astype("<f4").tobytes()).hexdigest()


def float64_sha256(array: numpy.ndarray) -> str:
    """Hash the little-endian bytes of a float64 array."""

    if not isinstance(array, numpy.ndarray) or array.dtype != numpy.float64:
        raise MetalFusedError("only float64 arrays are hashed here")
    return sha256(numpy.ascontiguousarray(array).astype("<f8").tobytes()).hexdigest()


def row_sum_errors(output: numpy.ndarray, layout: RowLayout) -> numpy.ndarray:
    """Return ``|sum(row) - 1|`` for each row of the packed outputs, in float64."""

    if (
        not isinstance(output, numpy.ndarray)
        or output.dtype != numpy.float32
        or output.shape != (layout.elements,)
    ):
        raise MetalFusedError("row sums need the packed float32 outputs of a layout")
    lengths = layout.lengths_array().astype(numpy.int64)
    starts = numpy.concatenate(([0], numpy.cumsum(lengths)[:-1]))
    return numpy.abs(numpy.add.reduceat(output.astype(numpy.float64), starts) - 1.0)
