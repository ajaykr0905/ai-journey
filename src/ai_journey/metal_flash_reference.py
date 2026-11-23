"""Seeded attention inputs, sequence layouts, float64 references and the bound.

A layout lists the length of each sequence, and ``gap`` guard rows follow
every sequence. A row holds ``HEAD_DIM = 64`` floats. Every gap row of Q, K
and V holds ``GAP_INPUT`` (``2**100``), so a kernel that reads one as a key
gets an infinite score and a NaN output. Every gap row of O starts as the
Day 79 sentinel, and a correct kernel keeps it. A timing layout packs its
sequences with no gap.

Every case draws from its own stream, ``numpy.random.default_rng([seed,
stream, index])``, so a run and the verifier rebuild identical inputs. The
draws come in the order Q, K, V.

The reference is float64 from the float32 inputs. The scores add the 64
products in column order. The exp is the Day 79 ``exp64``, which uses IEEE
operations and ``ldexp`` only. The weighted sums add the keys in order:
``numpy`` sums over the middle axis of a C-ordered array one key at a time,
and a test checks that order. So a report from one host verifies with the
same bits on another host.

The declared bound is a perturbation bound in ``a = sum_j P_j |v_j|``, the
softmax mean of ``|v|``, for each output value. It assumes the exp error
that the math probe measures, and a run stops when the probe breaks it.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy

from .kernel_metal_reference import MAX_SEED
from .metal_fused_reference import (
    FLOOR,
    FLUSH,
    GAP_INPUT,
    SENTINEL_BITS,
    UNIT,
    exp64,
    exp_growth,
    float32_sha256,
)
from .metal_flash_source import (
    BLOCK_KEYS,
    BLOCK_QUERIES,
    CAUSAL,
    HEAD_DIM,
    MODES,
    SCALE,
    MetalFlashError,
    require_size,
)

CORRECTNESS_STREAM = 0
BENCHMARK_STREAM = 1
STREAMS = (CORRECTNESS_STREAM, BENCHMARK_STREAM)
MAX_CASE_INDEX = 63
MAX_LENGTH = 8192
MAX_SEQUENCES = 512
MAX_BUFFER_ROWS = 1 << 18
MAX_SCORE_ELEMENTS = 1 << 28
MAX_GAP = 64
CORRECTNESS_GAP = 3
CHECK_ROWS = 32
REFERENCE_UNIT = 2.0**-53
# A float32 exp overflows above log(FLT_MAX).
FLOAT32_LOG_MAX = 88.72283905206835
# The bytes of one chunk of the weighted sums, so memory stays bounded.
CHUNK_BYTES = 1 << 25
VENDOR_UNIT = 2.0**-14
VENDOR_FLOOR = 2.0**-100

NORMAL = "normal"
WIDE_SCORES = "wide_scores"
OFFSET_VALUES = "offset_values"
DISTRIBUTIONS = {
    NORMAL: "q, k and v from a standard normal, drawn as float32",
    WIDE_SCORES: (
        "q and k from 5 times a standard normal and v from a standard normal, "
        "drawn in float64 and rounded to float32; the scores reach past 88.7, "
        "where a float32 exp with no max subtraction overflows"
    ),
    OFFSET_VALUES: (
        "q and k from a standard normal, drawn as float32, and v from 1000 plus "
        "a standard normal, rounded to float32"
    ),
}
REFERENCE = (
    "float64 attention of the float32 inputs: s = 0.125 * sum_c q_c k_c in "
    "column order, the max over the visible keys, w = exp64(s - max), and o = "
    "sum_j w_j v_j / sum_j w_j with the keys added in order; a causal query i "
    "sees the keys j <= i"
)
BOUND = (
    "|o - o_ref| <= a (2 E / (1 - E) (1 + u) + u + R) + 4 (L + 2 T + 2) 2**-126 "
    "vmax + 2**-125 for each output value, where a = sum_j P_j |v_jc|, vmax = "
    "max_j |v_jc| over the visible keys, E = (1 + A)(1 + gamma_(L + 2 T + 64)) - "
    "1, A = exp(D) (1 + EXP_ERROR)**(T + 1) (1 + u)**(2 T + 2) - 1, D = 2 (g + "
    "r) + 2 u (S + 2 g), g = gamma_66 * 0.125 |q| max_j |k_j|, r = the same "
    "with the unit 2**-53, S = max - min of the visible float64 scores, T = "
    "ceil(L / 32), u = 2**-24, EXP_ERROR = 4 * 2**-23 and R = 4 gamma_(L + "
    "70) for the unit 2**-53; one bound covers N0 and F1, because F1 has the "
    "more exp calls and roundings"
)
VENDOR_BOUND = (
    "|o - o_ref| <= 2**-14 a + 2**-100 for each output value of "
    "torch.nn.functional.scaled_dot_product_attention: a sanity bound only, "
    "because PyTorch names no kernel and no derived bound applies"
)


def gamma(count: int, unit: float = UNIT) -> float:
    """Return ``gamma_count = count u / (1 - count u)``."""

    product = count * unit
    return product / (1 - product)


def power(base: float, count: int) -> float:
    """Return ``base ** count`` by products, the same on every host."""

    value = 1.0
    for _ in range(count):
        value *= base
    return value


@dataclass(frozen=True)
class FlashLayout:
    """The sequence lengths of one batch and the guard rows after each one."""

    lengths: tuple[int, ...]
    gap: int

    def __post_init__(self) -> None:
        if (
            type(self.lengths) is not tuple
            or not 1 <= len(self.lengths) <= MAX_SEQUENCES
        ):
            raise MetalFlashError(f"a layout holds 1 to {MAX_SEQUENCES} sequences")
        for length in self.lengths:
            require_size(length, "sequence length", 1, MAX_LENGTH)
        require_size(self.gap, "gap", 0, MAX_GAP)
        if self.buffer_rows > MAX_BUFFER_ROWS:
            raise MetalFlashError(
                f"a layout of {self.buffer_rows} rows exceeds {MAX_BUFFER_ROWS}"
            )
        if self.score_elements > MAX_SCORE_ELEMENTS:
            raise MetalFlashError(
                f"a layout of {self.score_elements} scores exceeds "
                f"{MAX_SCORE_ELEMENTS}"
            )

    @property
    def sequences(self) -> int:
        """The number of sequences: batch times heads."""

        return len(self.lengths)

    @property
    def tokens(self) -> int:
        """The query rows of every sequence, gaps excluded."""

        return sum(self.lengths)

    @property
    def longest(self) -> int:
        """The longest sequence."""

        return max(self.lengths)

    @property
    def buffer_rows(self) -> int:
        """The rows of a Q, K, V or O buffer, gaps included."""

        return self.tokens + self.gap * self.sequences

    @property
    def buffer_elements(self) -> int:
        """The floats of a Q, K, V or O buffer."""

        return self.buffer_rows * HEAD_DIM

    @property
    def score_elements(self) -> int:
        """The floats of the N0 score buffer: L * L for each sequence."""

        return sum(length * length for length in self.lengths)

    @property
    def dense(self) -> bool:
        """Whether every sequence has one length."""

        return len(set(self.lengths)) == 1

    def blocks(self) -> int:
        """The F1 threadgroups: one for each block of 32 queries."""

        return sum(-(-length // BLOCK_QUERIES) for length in self.lengths)

    def pairs(self, mode: str) -> int:
        """The visible (query, key) pairs: L * L, or L (L + 1) / 2 if causal."""

        _mode(mode)
        if mode == CAUSAL:
            return sum(length * (length + 1) // 2 for length in self.lengths)
        return self.score_elements

    def tile_loads(self, mode: str) -> int:
        """The K/V tiles that the F1 groups load, over all query blocks."""

        _mode(mode)
        total = 0
        for length in self.lengths:
            tiles = -(-length // BLOCK_KEYS)
            blocks = -(-length // BLOCK_QUERIES)
            if mode == CAUSAL:
                total += sum(
                    -(-min(length, BLOCK_QUERIES * (block + 1)) // BLOCK_KEYS)
                    for block in range(blocks)
                )
            else:
                total += blocks * tiles
        return total

    def starts(self) -> numpy.ndarray:
        """The first buffer row of each sequence, as the uint32 array kernels read."""

        steps = numpy.asarray(self.lengths, dtype=numpy.int64) + self.gap
        return numpy.concatenate(([0], numpy.cumsum(steps)[:-1])).astype(numpy.uint32)

    def lengths_array(self) -> numpy.ndarray:
        """The sequence lengths, as the uint32 array the kernels read."""

        return numpy.asarray(self.lengths, dtype=numpy.uint32)

    def score_starts(self) -> numpy.ndarray:
        """The first score of each sequence in the N0 buffer, as uint32."""

        squares = numpy.asarray(self.lengths, dtype=numpy.int64) ** 2
        return numpy.concatenate(([0], numpy.cumsum(squares)[:-1])).astype(numpy.uint32)

    def query_groups(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """The sequence and row of each N0 threadgroup, one per query row."""

        lengths = numpy.asarray(self.lengths, dtype=numpy.int64)
        seqs = numpy.repeat(numpy.arange(self.sequences), lengths)
        rows = numpy.concatenate([numpy.arange(length) for length in self.lengths])
        return seqs.astype(numpy.uint32), rows.astype(numpy.uint32)

    def block_groups(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """The sequence and query block of each F1 threadgroup."""

        counts = [-(-length // BLOCK_QUERIES) for length in self.lengths]
        seqs = numpy.repeat(numpy.arange(self.sequences), counts)
        blocks = numpy.concatenate([numpy.arange(count) for count in counts])
        return seqs.astype(numpy.uint32), blocks.astype(numpy.uint32)

    def valid_rows(self) -> numpy.ndarray:
        """The buffer rows that hold a token, in sequence order."""

        starts = self.starts().astype(numpy.int64)
        return numpy.concatenate(
            [
                start + numpy.arange(length)
                for start, length in zip(starts, self.lengths)
            ]
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the layout as a report records it, the lengths by digest."""

        return {
            "sequences": self.sequences,
            "tokens": self.tokens,
            "shortest": min(self.lengths),
            "longest": self.longest,
            "gap": self.gap,
            "buffer_rows": self.buffer_rows,
            "score_elements": self.score_elements,
            "query_groups": self.tokens,
            "block_groups": self.blocks(),
            "dense": self.dense,
            "lengths_sha256": uint32_sha256(self.lengths_array()),
        }


def _mode(mode: str) -> str:
    if mode not in MODES:
        raise MetalFlashError(f"{mode!r} is not an attention mode")
    return mode


def dense_layout(sequences: int, length: int, gap: int = 0) -> FlashLayout:
    """A batch of ``sequences`` sequences of ``length`` tokens."""

    require_size(sequences, "sequences", 1, MAX_SEQUENCES)
    return FlashLayout((length,) * sequences, gap)


def uint32_sha256(array: numpy.ndarray) -> str:
    """Hash the little-endian bytes of a uint32 array."""

    if not isinstance(array, numpy.ndarray) or array.dtype != numpy.uint32:
        raise MetalFlashError("only uint32 arrays are hashed here")
    return sha256(numpy.ascontiguousarray(array).astype("<u4").tobytes()).hexdigest()


def float64_digest(array: numpy.ndarray) -> str:
    """Hash the little-endian bytes of a float64 array."""

    if not isinstance(array, numpy.ndarray) or array.dtype != numpy.float64:
        raise MetalFlashError("only float64 arrays are hashed here")
    return sha256(numpy.ascontiguousarray(array).astype("<f8").tobytes()).hexdigest()


def check_rows(length: int, count: int | None) -> numpy.ndarray:
    """The query rows of one sequence that a check compares: all, or ``count``.

    ``count`` rows spread evenly from row 0 to row L - 1 keep both ends, so a
    timing point checks the first query, the last query and the rows between.
    """

    require_size(length, "sequence length", 1, MAX_LENGTH)
    if count is None or count >= length:
        return numpy.arange(length, dtype=numpy.int64)
    require_size(count, "check rows", 2, MAX_LENGTH)
    return numpy.unique(numpy.rint(numpy.linspace(0, length - 1, count))).astype(
        numpy.int64
    )


@dataclass(frozen=True)
class FlashInputs:
    """The Q, K and V buffers of one case or point, with gap rows."""

    layout: FlashLayout
    q: numpy.ndarray
    k: numpy.ndarray
    v: numpy.ndarray

    def digests(self) -> dict[str, str]:
        """Hash the three inputs as a report records them."""

        return {
            "q_sha256": float32_sha256(self.q),
            "k_sha256": float32_sha256(self.k),
            "v_sha256": float32_sha256(self.v),
        }

    def sequence(self, index: int) -> tuple[numpy.ndarray, ...]:
        """Return the float32 Q, K and V rows of one sequence, as L x 64 arrays."""

        start = int(self.layout.starts()[index])
        stop = start + self.layout.lengths[index]
        return tuple(
            array.reshape(-1, HEAD_DIM)[start:stop]
            for array in (self.q, self.k, self.v)
        )


def _draw(generator, count: int, distribution: str):
    if distribution == WIDE_SCORES:
        q = (5.0 * generator.standard_normal(count)).astype(numpy.float32)
        k = (5.0 * generator.standard_normal(count)).astype(numpy.float32)
        return q, k, generator.standard_normal(count, dtype=numpy.float32)
    q = generator.standard_normal(count, dtype=numpy.float32)
    k = generator.standard_normal(count, dtype=numpy.float32)
    if distribution == OFFSET_VALUES:
        return q, k, (1000.0 + generator.standard_normal(count)).astype(numpy.float32)
    return q, k, generator.standard_normal(count, dtype=numpy.float32)


def flash_inputs(
    seed: int, stream: int, index: int, layout: FlashLayout, distribution: str
) -> FlashInputs:
    """Draw one case's or point's Q, K and V from its own stream."""

    if not isinstance(layout, FlashLayout):
        raise TypeError("layout must be a FlashLayout")
    if distribution not in DISTRIBUTIONS:
        raise MetalFlashError(f"distribution must be one of {', '.join(DISTRIBUTIONS)}")
    require_size(seed, "seed", 0, MAX_SEED)
    if stream not in STREAMS:
        raise MetalFlashError(f"stream must be one of {STREAMS}")
    require_size(index, "case index", 0, MAX_CASE_INDEX)
    generator = numpy.random.default_rng([seed, stream, index])
    drawn = _draw(generator, layout.tokens * HEAD_DIM, distribution)
    if layout.gap == 0:
        return FlashInputs(layout, *drawn)
    valid = layout.valid_rows()
    buffers = []
    for values in drawn:
        buffer = numpy.full((layout.buffer_rows, HEAD_DIM), GAP_INPUT, numpy.float32)
        buffer[valid] = values.reshape(-1, HEAD_DIM)
        buffers.append(buffer.reshape(-1))
    return FlashInputs(layout, *buffers)


def sentinel_output(layout: FlashLayout) -> numpy.ndarray:
    """Return an O buffer of quiet-NaN sentinels with fixed bits."""

    return numpy.full(layout.buffer_elements, SENTINEL_BITS, dtype=numpy.uint32).view(
        numpy.float32
    )


@dataclass(frozen=True)
class FlashReference:
    """The float64 output of the checked rows, with the bound of each value.

    ``rows`` lists the checked buffer rows in sequence order. ``output``,
    ``bound`` and ``vendor_bound`` hold one row of 64 values for each.
    """

    mode: str
    rows: numpy.ndarray
    output: numpy.ndarray
    bound: numpy.ndarray
    vendor_bound: numpy.ndarray
    max_score: float


def _norms(rows: numpy.ndarray) -> numpy.ndarray:
    total = numpy.zeros(rows.shape[0], dtype=numpy.float64)
    for column in range(HEAD_DIM):
        total = total + rows[:, column] * rows[:, column]
    return numpy.sqrt(total)


def _scores(q: numpy.ndarray, k: numpy.ndarray) -> numpy.ndarray:
    total = numpy.zeros((q.shape[0], k.shape[0]), dtype=numpy.float64)
    for column in range(HEAD_DIM):
        total = total + q[:, column, None] * k[None, :, column]
    return total * SCALE


def _sequence(q, k, v, queries: numpy.ndarray, mode: str):
    """Return the output, the bound and the largest score of some query rows."""

    length = k.shape[0]
    tiles = -(-length // BLOCK_KEYS)
    q64, k64, v64 = (array.astype(numpy.float64) for array in (q, k, v))
    key_norms = _norms(k64)
    magnitudes = numpy.abs(v64)
    causal = mode == CAUSAL
    if causal:
        norm_limit = numpy.maximum.accumulate(key_norms)[queries]
        value_limit = numpy.maximum.accumulate(magnitudes, axis=0)[queries]
    else:
        norm_limit = numpy.full(queries.size, key_norms.max())
        value_limit = numpy.repeat(magnitudes.max(axis=0)[None], queries.size, 0)
    extended = numpy.concatenate((v64, magnitudes, numpy.ones((length, 1))), axis=1)
    width = extended.shape[1]
    step = max(1, CHUNK_BYTES // (8 * length * width))
    outputs, means, tops, spreads = [], [], [], []
    for first in range(0, queries.size, step):
        chosen = queries[first : first + step]
        scores = _scores(q64[chosen], k64)
        visible = numpy.ones(scores.shape, dtype=bool)
        if causal:
            visible = numpy.arange(length)[None, :] <= chosen[:, None]
        top = numpy.where(visible, scores, -numpy.inf).max(axis=1)
        bottom = numpy.where(visible, scores, numpy.inf).min(axis=1)
        shifted = numpy.where(visible, scores - top[:, None], 0.0)
        weights = numpy.where(visible, exp64(shifted), 0.0)
        sums = (weights[:, :, None] * extended[None, :, :]).sum(axis=1)
        outputs.append(sums[:, :HEAD_DIM] / sums[:, -1:])
        means.append(sums[:, HEAD_DIM:-1] / sums[:, -1:])
        tops.append(top)
        spreads.append(top - bottom)
    output = numpy.concatenate(outputs)
    mean = numpy.concatenate(means)
    spread = numpy.concatenate(spreads)
    products = SCALE * _norms(q64[queries]) * norm_limit
    g = gamma(HEAD_DIM + 2) * products
    r = gamma(HEAD_DIM + 2, REFERENCE_UNIT) * products
    shift = 2.0 * (g + r) + 2.0 * UNIT * (spread + 2.0 * g)
    growth = exp_growth(tiles + 1) * power(1.0 + UNIT, 2 * tiles + 2)
    a = exp64(shift) * growth - 1.0
    e = (1.0 + a) * (1.0 + gamma(length + 2 * tiles + 64)) - 1.0
    if not numpy.all(e < 0.5):
        raise MetalFlashError("the attention bound is too weak to use")
    relative = 2.0 * e / (1.0 - e) * (1.0 + UNIT) + UNIT
    relative = relative + 4.0 * gamma(length + 70, REFERENCE_UNIT)
    tail = 4.0 * (length + 2 * tiles + 2) * FLUSH * value_limit + FLOOR
    bound = mean * relative[:, None] + tail
    vendor = VENDOR_UNIT * mean + VENDOR_FLOOR
    return output, bound, vendor, float(numpy.concatenate(tops).max())


def flash_reference(
    inputs: FlashInputs, mode: str, count: int | None = None
) -> FlashReference:
    """Compute the float64 output of the checked rows, with their bounds.

    ``count=None`` checks every row; a count checks that many rows of each
    sequence, spread by ``check_rows``.
    """

    if not isinstance(inputs, FlashInputs):
        raise TypeError("inputs must be FlashInputs")
    _mode(mode)
    layout = inputs.layout
    rows, outputs, bounds, vendors, tops = [], [], [], [], []
    for index, (start, length) in enumerate(zip(layout.starts(), layout.lengths)):
        q, k, v = inputs.sequence(index)
        if not all(numpy.all(numpy.isfinite(array)) for array in (q, k, v)):
            raise MetalFlashError("every input must be finite")
        queries = check_rows(length, count)
        output, bound, vendor, top = _sequence(q, k, v, queries, mode)
        rows.append(int(start) + queries)
        outputs.append(output)
        bounds.append(bound)
        vendors.append(vendor)
        tops.append(top)
    return FlashReference(
        mode=mode,
        rows=numpy.concatenate(rows),
        output=numpy.concatenate(outputs),
        bound=numpy.concatenate(bounds),
        vendor_bound=numpy.concatenate(vendors),
        max_score=max(tops),
    )


def unscaled_exp_overflows(reference: FlashReference) -> bool:
    """Whether a float32 exp of the largest score, with no max subtracted, overflows.

    Above about 88.7 it does, which shows that a finite softmax needs the
    max subtraction of N0 or the running max of F1.
    """

    return not reference.max_score < FLOAT32_LOG_MAX


def gather_rows(buffer: numpy.ndarray, rows: numpy.ndarray) -> numpy.ndarray:
    """Return the checked rows of a flat O buffer as an r x 64 array."""

    return buffer.reshape(-1, HEAD_DIM)[rows]
