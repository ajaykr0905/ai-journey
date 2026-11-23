"""Seeded inputs, row layouts, float64 references and the declared error bounds.

A layout lists the length of each row, and ``gap`` guard slots follow every
row. Every gap slot of an input holds ``GAP_INPUT`` (``2**100``), so a
kernel that reads one gets a wrong sum of squares or a wrong max. Every gap
slot of an output starts as the Day 79 sentinel, and a correct kernel keeps
it. The cross-entropy output holds one loss for each row and then ``gap``
sentinel slots. A timing layout packs its rows with no gap.

Every case draws from its own stream, ``numpy.random.default_rng([seed,
family, stream, index])``, so a run and the verifier rebuild identical
inputs. The first draw takes the main input of the named Day 79
distribution: ``x`` for RMSNorm and cross-entropy, ``a`` for SwiGLU. Then
RMSNorm draws ``w`` (one standard normal for each column of the longest
row), SwiGLU draws ``b`` (a standard normal), and cross-entropy draws one
target column for each row, uniform over the row.

Each reference is float64 from the float32 inputs. Its exp is the Day 79
``exp64`` and its log is ``log64``: both use only IEEE operations and
``frexp`` or ``ldexp``, so a report from one host verifies with the same
bits on another host. ``u = 2**-24`` is the unit roundoff of float32, and
``gamma_k = k u / (1 - k u)``. Each bound assumes the error of one
``precise::`` call that the math probe measures; a run stops when a probe
breaks its assumption.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256

import numpy

from .kernel_metal_reference import MAX_SEED
from .metal_fused_reference import (
    DISTRIBUTIONS,
    EXP_ERROR,
    FLOOR,
    FLUSH,
    GAP_INPUT,
    SENTINEL_BITS,
    UNIT,
    draw_values,
    exp64,
    exp_growth,
    float32_sha256,
)
from .metal_fused_source import SIMD_WIDTH
from .metal_liger_source import (
    CROSS_ENTROPY,
    EPS,
    FAMILIES,
    MAX_STREAM_ROW,
    RMSNORM,
    SWIGLU,
    MetalLigerError,
    columns_per_thread,
    launch_group,
    require_size,
)

CORRECTNESS_STREAM = 0
BENCHMARK_STREAM = 1
STREAMS = (CORRECTNESS_STREAM, BENCHMARK_STREAM)
FAMILY_CODES = {RMSNORM: 0, SWIGLU: 1, CROSS_ENTROPY: 2}
MAX_CASE_INDEX = 63
MAX_ROWS = 1 << 16
MAX_ELEMENTS = 1 << 26
MAX_GAP = 1024
CORRECTNESS_GAP = 32
CHUNK_ELEMENTS = 1 << 22
EPS32 = float(numpy.float32(EPS))
REFERENCE_UNIT = 2.0**-53
RSQRT_ULPS = 2
RSQRT_ERROR = RSQRT_ULPS * 2.0**-23
LOG_ULPS = 4
LOG_ERROR = LOG_ULPS * 2.0**-23
SWIGLU_TAIL = 2.0**-120
SWIGLU_REFERENCE = 16 * REFERENCE_UNIT
VENDOR_UNIT = 2.0**-16
VENDOR_FLOOR = 2.0**-100
REFERENCE = {
    RMSNORM: (
        "float64 y = x * w / sqrt(sum(x^2) / n + eps) with numpy.add.reduceat "
        "over each row and eps = float32(1e-6)"
    ),
    SWIGLU: (
        "float64 y = silu(a) * b with silu(a) = a / (1 + exp64(-a)) for a >= 0 "
        "and a * exp64(a) / (1 + exp64(a)) for a < 0"
    ),
    CROSS_ENTROPY: (
        "float64 loss = (max - x[t]) + log64(sum exp64(x - max)) for each row, "
        "with numpy.add.reduceat; exp64 and log64 use IEEE operations, frexp "
        "and ldexp only, so each host computes the same bits"
    ),
}
BOUNDS = {
    RMSNORM: (
        "|y - y_ref| <= (rel + ref) / (1 - ref) * |y_ref| + 2**-125 with rel = "
        "(1 + c) (1 + RSQRT_ERROR) (1 + u)**2 - 1, c = 1 / sqrt(1 - g) - 1, g = "
        "gamma_(A + 2), A = k + 31 + (G / 32 - 1) fma and add roundings of the "
        "sum of squares, k = ceil(n / G), RSQRT_ERROR = 2 * 2**-23 and ref = "
        "gamma_(n + 8) for the unit 2**-53"
    ),
    SWIGLU: (
        "|y - y_ref| <= (rel + ref) / (1 - ref) * |y_ref| + 2**-120 |b| + 2**-125 "
        "with rel = (1 + e / (1 - e)) (1 + u)**2 - 1, e = (1 + EXP_ERROR) (1 + u) "
        "- 1, EXP_ERROR = 4 * 2**-23 and ref = 16 * 2**-53; the 2**-120 |b| term "
        "covers exp(-a) overflow for a below -88.7, where the kernel returns 0"
    ),
    CROSS_ENTROPY: (
        "|loss - loss_ref| <= i + u (loss_ref + i) + R + 2**-125 with i = u m + t "
        "+ LOG_ERROR (log S + t), m = max - x[t], t = rho / (1 - rho), rho = "
        "exp(N u D) (1 + EXP_ERROR)**N (1 + gamma_A) - 1 + (n + 2 G) 2**-126, D = "
        "max - min of the row, k = ceil(n / G), N = k + 2, A = k + 1 + 31 + 1 + "
        "(G / 32 - 1), LOG_ERROR = 4 * 2**-23 and R = gamma_(n + 8) (1 + 2 "
        "loss_ref) for the unit 2**-53"
    ),
}
VENDOR_BOUNDS = {
    RMSNORM: "|y - y_ref| <= 2**-16 |y_ref| + 2**-100: a sanity bound only",
    SWIGLU: "|y - y_ref| <= 2**-16 |y_ref| + 2**-100: a sanity bound only",
    CROSS_ENTROPY: "|loss - loss_ref| <= 2**-16 (loss_ref + 1): a sanity bound only",
}

_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_SQRT_HALF = 0.70710678118654757274
# 1/1, 1/3, 1/5, ...: the series of atanh(f) / f in f**2, for |f| <= 0.1716.
_ATANH = tuple(1.0 / (2 * order + 1) for order in range(14))


def log64(values: object) -> numpy.ndarray:
    """Return log of positive finite float64 values, the same on every host.

    ``numpy.log`` calls the libm of the platform, so two hosts can round one
    result to different neighbors. Here ``x = m 2**e`` with ``m`` from
    ``sqrt(1/2)`` to ``sqrt(2)`` by ``frexp``, and ``log m = 2 atanh(f)`` with
    ``f = (m - 1) / (m + 1)``; each step is an IEEE operation.
    """

    x = numpy.asarray(values, dtype=numpy.float64)
    if not numpy.all(numpy.isfinite(x)) or not numpy.all(x > 0):
        raise MetalLigerError("log64 takes positive finite values only")
    mantissa, exponent = numpy.frexp(x)
    low = mantissa < _SQRT_HALF
    mantissa = numpy.where(low, mantissa * 2.0, mantissa)
    exponent = numpy.where(low, exponent - 1, exponent).astype(numpy.float64)
    f = (mantissa - 1.0) / (mantissa + 1.0)
    square = f * f
    series = numpy.full_like(f, _ATANH[-1])
    for term in reversed(_ATANH[:-1]):
        series = series * square + term
    return exponent * _LN2_HI + (exponent * _LN2_LO + 2.0 * f * series)


def gamma(count: int, unit: float = UNIT) -> float:
    """Return ``gamma_count = count u / (1 - count u)``."""

    product = count * unit
    return product / (1 - product)


@dataclass(frozen=True)
class LigerLayout:
    """The row lengths of one batch and the guard slots after each row."""

    lengths: tuple[int, ...]
    gap: int

    def __post_init__(self) -> None:
        if type(self.lengths) is not tuple or not 1 <= len(self.lengths) <= MAX_ROWS:
            raise MetalLigerError(f"a layout holds 1 to {MAX_ROWS} rows")
        for length in self.lengths:
            require_size(length, "row length", 1, MAX_STREAM_ROW)
        require_size(self.gap, "gap", 0, MAX_GAP)
        if self.buffer_elements > MAX_ELEMENTS:
            raise MetalLigerError(
                f"a layout of {self.buffer_elements} slots exceeds {MAX_ELEMENTS}"
            )

    @property
    def rows(self) -> int:
        """The number of rows, one threadgroup each."""

        return len(self.lengths)

    @property
    def elements(self) -> int:
        """The elements of every row, gaps excluded."""

        return sum(self.lengths)

    @property
    def longest(self) -> int:
        """The longest row, which sets the group size."""

        return max(self.lengths)

    @property
    def buffer_elements(self) -> int:
        """The slots of an input buffer, gaps included."""

        return self.elements + self.gap * self.rows

    @property
    def dense(self) -> bool:
        """Whether every row has one length."""

        return len(set(self.lengths)) == 1

    def group_size(self, family: str) -> int:
        """The launch rule's G for this batch and one family."""

        return launch_group(family, self.longest)

    def starts(self) -> numpy.ndarray:
        """The first slot of each row, as the uint32 array the kernels read."""

        steps = numpy.asarray(self.lengths, dtype=numpy.int64) + self.gap
        return numpy.concatenate(([0], numpy.cumsum(steps)[:-1])).astype(numpy.uint32)

    def lengths_array(self) -> numpy.ndarray:
        """The row lengths, as the uint32 array the kernels read."""

        return numpy.asarray(self.lengths, dtype=numpy.uint32)

    def packed_starts(self) -> numpy.ndarray:
        """The first packed element of each row, as int64."""

        lengths = numpy.asarray(self.lengths, dtype=numpy.int64)
        return numpy.concatenate(([0], numpy.cumsum(lengths)[:-1]))

    def valid(self) -> numpy.ndarray:
        """The buffer slots that hold a row element, in row order."""

        if self.gap == 0:
            return numpy.arange(self.elements, dtype=numpy.int64)
        lengths = numpy.asarray(self.lengths, dtype=numpy.int64)
        shift = self.starts().astype(numpy.int64) - self.packed_starts()
        return numpy.arange(self.elements, dtype=numpy.int64) + numpy.repeat(
            shift, lengths
        )

    def output_elements(self, family: str) -> int:
        """The slots of the output buffer: the input's, or one loss per row."""

        if family == CROSS_ENTROPY:
            return self.rows + self.gap
        return self.buffer_elements

    def output_valid(self, family: str) -> numpy.ndarray:
        """The output slots that a correct kernel writes, in row order."""

        if family == CROSS_ENTROPY:
            return numpy.arange(self.rows, dtype=numpy.int64)
        return self.valid()

    def chunks(self, limit: int = CHUNK_ELEMENTS) -> Iterator[tuple[int, int]]:
        """Split the rows into runs of at most ``limit`` elements, or one row."""

        first, total = 0, 0
        for row, length in enumerate(self.lengths):
            if row > first and total + length > limit:
                yield first, row
                first, total = row, 0
            total += length
        yield first, self.rows

    def to_mapping(self, family: str) -> dict[str, object]:
        """Return the layout as a report records it, the lengths by digest."""

        return {
            "rows": self.rows,
            "elements": self.elements,
            "shortest": min(self.lengths),
            "longest": self.longest,
            "group_size": self.group_size(family),
            "gap": self.gap,
            "buffer_elements": self.buffer_elements,
            "output_elements": self.output_elements(family),
            "dense": self.dense,
            "lengths_sha256": uint32_sha256(self.lengths_array()),
        }


def dense_layout(rows: int, cols: int, gap: int = 0) -> LigerLayout:
    """A batch of ``rows`` rows of ``cols`` elements."""

    require_size(rows, "rows", 1, MAX_ROWS)
    return LigerLayout((cols,) * rows, gap)


def uint32_sha256(array: numpy.ndarray) -> str:
    """Hash the little-endian bytes of a uint32 array."""

    if not isinstance(array, numpy.ndarray) or array.dtype != numpy.uint32:
        raise MetalLigerError("only uint32 arrays are hashed here")
    return sha256(numpy.ascontiguousarray(array).astype("<u4").tobytes()).hexdigest()


def _generator(seed: int, family: str, stream: int, index: int):
    require_size(seed, "seed", 0, MAX_SEED)
    if family not in FAMILIES:
        raise MetalLigerError(f"{family!r} is not a kernel family")
    if stream not in STREAMS:
        raise MetalLigerError(f"stream must be one of {STREAMS}")
    require_size(index, "case index", 0, MAX_CASE_INDEX)
    return numpy.random.default_rng([seed, FAMILY_CODES[family], stream, index])


@dataclass(frozen=True)
class LigerInputs:
    """The device inputs of one case or point: a main buffer and a second one.

    ``main`` is ``x`` or ``a`` with its gaps. ``second`` is ``w`` (one float
    for each column of the longest row), ``b`` (with gaps) or the uint32
    targets (one for each row).
    """

    family: str
    layout: LigerLayout
    main: numpy.ndarray
    second: numpy.ndarray

    def digests(self) -> dict[str, str]:
        """Hash both inputs as a report records them."""

        second = (
            uint32_sha256(self.second)
            if self.family == CROSS_ENTROPY
            else float32_sha256(self.second)
        )
        return {"main_sha256": float32_sha256(self.main), "second_sha256": second}


def _gapped(values: numpy.ndarray, layout: LigerLayout) -> numpy.ndarray:
    if layout.gap == 0:
        return values
    buffer = numpy.full(layout.buffer_elements, GAP_INPUT, dtype=numpy.float32)
    buffer[layout.valid()] = values
    return buffer


def liger_inputs(
    seed: int,
    family: str,
    stream: int,
    index: int,
    layout: LigerLayout,
    distribution: str,
) -> LigerInputs:
    """Draw one case's or point's inputs from its own stream."""

    if not isinstance(layout, LigerLayout):
        raise TypeError("layout must be a LigerLayout")
    if distribution not in DISTRIBUTIONS:
        raise MetalLigerError(f"distribution must be one of {', '.join(DISTRIBUTIONS)}")
    generator = _generator(seed, family, stream, index)
    main = _gapped(draw_values(generator, layout.elements, distribution), layout)
    if family == RMSNORM:
        second = generator.standard_normal(layout.longest, dtype=numpy.float32)
    elif family == SWIGLU:
        second = _gapped(
            generator.standard_normal(layout.elements, dtype=numpy.float32), layout
        )
    else:
        lengths = numpy.asarray(layout.lengths, dtype=numpy.int64)
        second = generator.integers(0, lengths, dtype=numpy.int64).astype(numpy.uint32)
    return LigerInputs(family, layout, main, second)


def sentinel_output(family: str, layout: LigerLayout) -> numpy.ndarray:
    """Return an output buffer of quiet-NaN sentinels with fixed bits."""

    return numpy.full(
        layout.output_elements(family), SENTINEL_BITS, dtype=numpy.uint32
    ).view(numpy.float32)


@dataclass(frozen=True)
class LigerReference:
    """The float64 output of one case or point, with the bound of each value.

    ``output`` and ``bound`` are packed: the row elements in row order, or
    one loss for each row.
    """

    family: str
    output: numpy.ndarray
    bound: numpy.ndarray
    vendor_bound: numpy.ndarray


def _row_slices(layout: LigerLayout, first: int, last: int) -> tuple[int, int]:
    starts = layout.packed_starts()
    end = layout.elements if last == layout.rows else int(starts[last])
    return int(starts[first]), end


def _rmsnorm(inputs: LigerInputs) -> tuple[numpy.ndarray, numpy.ndarray]:
    layout = inputs.layout
    x = inputs.main[layout.valid()].astype(numpy.float64)
    lengths = numpy.asarray(layout.lengths, dtype=numpy.int64)
    starts = layout.packed_starts()
    columns = numpy.arange(layout.elements, dtype=numpy.int64) - numpy.repeat(
        starts, lengths
    )
    w = inputs.second.astype(numpy.float64)[columns]
    squares = numpy.add.reduceat(x * x, starts)
    inverse = 1.0 / numpy.sqrt(squares / lengths + EPS32)
    output = x * numpy.repeat(inverse, lengths) * w
    size = layout.group_size(RMSNORM)
    rel = []
    for length in layout.lengths:
        roundings = columns_per_thread(length, size) + 31 + (size // SIMD_WIDTH - 1)
        g = gamma(roundings + 2)
        c = 1.0 / math.sqrt(1.0 - g) - 1.0
        rel.append((1 + c) * (1 + RSQRT_ERROR) * (1 + UNIT) * (1 + UNIT) - 1)
    ref = numpy.array([gamma(length + 8, REFERENCE_UNIT) for length in lengths])
    rel_each = numpy.repeat(numpy.array(rel), lengths)
    ref_each = numpy.repeat(ref, lengths)
    bound = (rel_each + ref_each) / (1 - ref_each) * numpy.abs(output) + FLOOR
    return output, bound


def silu64(values: numpy.ndarray) -> numpy.ndarray:
    """Return float64 silu by IEEE operations and exp64, stable at both signs."""

    a = numpy.asarray(values, dtype=numpy.float64)
    negative = a < 0
    e = exp64(numpy.where(negative, a, -a))
    return numpy.where(negative, a * e / (1.0 + e), a / (1.0 + e))


def _swiglu(inputs: LigerInputs) -> tuple[numpy.ndarray, numpy.ndarray]:
    layout = inputs.layout
    valid = layout.valid()
    a = inputs.main[valid].astype(numpy.float64)
    b = inputs.second[valid].astype(numpy.float64)
    output = numpy.empty(layout.elements, dtype=numpy.float64)
    for first, last in layout.chunks():
        low, high = _row_slices(layout, first, last)
        output[low:high] = silu64(a[low:high]) * b[low:high]
    e = (1 + EXP_ERROR) * (1 + UNIT) - 1
    rel = (1 + e / (1 - e)) * (1 + UNIT) * (1 + UNIT) - 1
    ref = SWIGLU_REFERENCE
    bound = (rel + ref) / (1 - ref) * numpy.abs(output)
    bound = bound + SWIGLU_TAIL * numpy.abs(b) + FLOOR
    return output, bound


def _cross_entropy(inputs: LigerInputs) -> tuple[numpy.ndarray, numpy.ndarray]:
    layout = inputs.layout
    valid = layout.valid()
    targets = inputs.second.astype(numpy.int64)
    starts = layout.packed_starts()
    size = layout.group_size(CROSS_ENTROPY)
    losses = numpy.empty(layout.rows, dtype=numpy.float64)
    bounds = numpy.empty(layout.rows, dtype=numpy.float64)
    for first, last in layout.chunks():
        low, high = _row_slices(layout, first, last)
        x = inputs.main[valid[low:high]].astype(numpy.float64)
        local = starts[first:last] - low
        lengths = numpy.asarray(layout.lengths[first:last], dtype=numpy.int64)
        top = numpy.maximum.reduceat(x, local)
        bottom = numpy.minimum.reduceat(x, local)
        sums = numpy.add.reduceat(exp64(x - numpy.repeat(top, lengths)), local)
        log_sum = log64(sums)
        margin = top - x[local + targets[first:last]]
        loss = margin + log_sum
        losses[first:last] = loss
        for offset, length in enumerate(layout.lengths[first:last]):
            k = columns_per_thread(length, size)
            calls = k + 2
            roundings = k + 1 + 31 + 1 + (size // SIMD_WIDTH - 1)
            spread = float(top[offset] - bottom[offset])
            shift = float(exp64(numpy.array([calls * UNIT * spread]))[0])
            rho = shift * exp_growth(calls) * (1 + gamma(roundings)) - 1
            rho += (length + 2 * size) * FLUSH
            if not rho < 0.5:
                raise MetalLigerError("the cross-entropy sum bound is too weak to use")
            t = rho / (1 - rho)
            inner = UNIT * float(margin[offset]) + t
            inner += LOG_ERROR * (float(log_sum[offset]) + t)
            value = float(loss[offset])
            ref = gamma(length + 8, REFERENCE_UNIT) * (1 + 2 * value)
            bounds[first + offset] = inner + UNIT * (value + inner) + ref + FLOOR
    return losses, bounds


def liger_reference(inputs: LigerInputs) -> LigerReference:
    """Compute the float64 output of one case or point, with its bounds."""

    if not isinstance(inputs, LigerInputs):
        raise TypeError("inputs must be LigerInputs")
    layout = inputs.layout
    values = inputs.main[layout.valid()]
    if not numpy.all(numpy.isfinite(values)):
        raise MetalLigerError("every input must be finite")
    if inputs.family == RMSNORM:
        output, bound = _rmsnorm(inputs)
    elif inputs.family == SWIGLU:
        output, bound = _swiglu(inputs)
    else:
        output, bound = _cross_entropy(inputs)
    if inputs.family == CROSS_ENTROPY:
        vendor = VENDOR_UNIT * (output + 1.0)
    else:
        vendor = VENDOR_UNIT * numpy.abs(output) + VENDOR_FLOOR
    return LigerReference(inputs.family, output, bound, vendor)


def naive_cross_entropy_finite(inputs: LigerInputs) -> bool:
    """Whether log(sum(exp(x))) - x[t] in float32, with no max, is finite.

    For logits near +1000 it is not, which shows that a finite loss needs
    the max subtraction or the online rescaling.
    """

    layout = inputs.layout
    if inputs.family != CROSS_ENTROPY:
        raise MetalLigerError("the naive check belongs to cross-entropy")
    x = inputs.main[layout.valid()]
    starts = layout.packed_starts()
    with numpy.errstate(all="ignore"):
        sums = numpy.add.reduceat(numpy.exp(x), starts, dtype=numpy.float32)
        loss = numpy.log(sums) - x[starts + inputs.second.astype(numpy.int64)]
    return bool(numpy.all(numpy.isfinite(loss)))


PROBE_COUNT = 1 << 16
PROBE_RULES = {
    "exp": (
        "each result whose float64 exp is at least 2**-126 has a relative error "
        "of at most EXP_ERROR = 4 * 2**-23; each smaller result is within 2**-126"
    ),
    "rsqrt": "each result has a relative error of at most RSQRT_ERROR = 2 * 2**-23",
    "log": "each result has an error of at most LOG_ERROR |log x|, 4 * 2**-23 |log x|",
}
PROBE_ARGUMENTS = {
    "exp": "the Day 79 exp probe: 2**18 float32 arguments from -104 to 0 and extremes",
    "rsqrt": "2**16 float32 arguments, 2**(-20 + 60 i / (2**16 - 1)) for each i",
    "log": (
        "2**15 float32 arguments evenly spaced from 1 to 2**17, then 1 + i 2**-23 "
        "for i below 2**15"
    ),
}


def rsqrt_probe_arguments() -> numpy.ndarray:
    """Return the rsqrt probe arguments, over the range of mean(x^2) + eps."""

    return numpy.exp2(numpy.linspace(-20.0, 40.0, PROBE_COUNT)).astype(numpy.float32)


def log_probe_arguments() -> numpy.ndarray:
    """Return the log probe arguments, over the range of a row's sum of exp."""

    half = PROBE_COUNT // 2
    spread = numpy.linspace(1.0, 2.0**17, half)
    near = 1.0 + numpy.arange(half, dtype=numpy.float64) * 2.0**-23
    return numpy.concatenate((spread, near)).astype(numpy.float32)


@dataclass(frozen=True)
class MathProbe:
    """The measured relative error of one set of ``precise::`` results."""

    function: str
    results_sha256: str
    max_relative_error: float
    max_ulps: float
    exact_zero: bool | None

    @property
    def assumed_error(self) -> float:
        """The relative error that the bounds assume for this function."""

        return RSQRT_ERROR if self.function == "rsqrt" else LOG_ERROR

    @property
    def holds(self) -> bool:
        """Whether the results meet the declared rule of the function."""

        return self.max_relative_error <= self.assumed_error and (
            self.exact_zero is not False
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the measured errors as a report records them."""

        return {
            "function": self.function,
            "count": PROBE_COUNT,
            "arguments": PROBE_ARGUMENTS[self.function],
            "rule": PROBE_RULES[self.function],
            "assumed_error": self.assumed_error,
            "results_sha256": self.results_sha256,
            "max_relative_error": self.max_relative_error,
            "max_ulps": self.max_ulps,
            "exact_zero": self.exact_zero,
            "holds": self.holds,
        }


def math_probe_errors(function: str, results: numpy.ndarray) -> MathProbe:
    """Measure rsqrt or log results against their float64 values."""

    if function == "rsqrt":
        arguments = rsqrt_probe_arguments()
        exact = 1.0 / numpy.sqrt(arguments.astype(numpy.float64))
    elif function == "log":
        arguments = log_probe_arguments()
        exact = log64(arguments.astype(numpy.float64))
    else:
        raise MetalLigerError(f"{function!r} has no probe here")
    if (
        not isinstance(results, numpy.ndarray)
        or results.dtype != numpy.float32
        or results.shape != arguments.shape
    ):
        raise MetalLigerError("a probe must return one float32 per argument")
    got = results.astype(numpy.float64)
    if not numpy.all(numpy.isfinite(got)):
        raise MetalLigerError("a probe returned a value that is not finite")
    error = numpy.abs(got - exact)
    nonzero = exact != 0
    relative = error[nonzero] / numpy.abs(exact[nonzero])
    spacing = numpy.spacing(numpy.abs(exact[nonzero]).astype(numpy.float32))
    zero = None
    if function == "log":
        zero = bool(numpy.all(got[~nonzero] == 0.0))
    return MathProbe(
        function=function,
        results_sha256=float32_sha256(results),
        max_relative_error=float(relative.max(initial=0.0)),
        max_ulps=float((error[nonzero] / spacing.astype(numpy.float64)).max()),
        exact_zero=zero,
    )
