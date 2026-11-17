"""Seeded inputs and exact CPU rebuilds of every ceiling kernel's output.

Each input comes from its own stream, ``numpy.random.default_rng([seed,
stream, index])``, so a run and an independent verifier rebuild the same
inputs from the seed alone.

FMA chains start from uniform float32 values in [1, 2): ``m * 2**-23`` for
integers ``m`` in [2**23, 2**24). The multiplier ``a = (2**24 - 4173) *
2**-24`` and the addend ``b = 12561923 * 2**-35`` are exact float32 values
with full 24-bit significands, so a fused and an unfused multiply-add round
differently in most steps and the check can tell them apart. For ``x`` in
[1, 2) the exact ``x * a + b`` is an integer multiple of 2**-47 below 4, so it
has at most 49 significant bits and float64 holds it exactly; rounding that
float64 to float32 is then the one rounding of a fused multiply-add. Each
step moves ``x`` toward ``b / (1 - a)``, about 1.47, so ``x`` never leaves
[1, 2) and the argument holds at every step. One thread's output is the XOR
of its chains' final bit patterns, and thread ``i`` uses row ``i % 32`` of
the start table, so 32 words rebuild a whole output buffer.

Copy sources are standard-normal float32 values, so every bit of a copy
matters. Read-reduce sources are whole numbers in [0, 1024), so a partial of
at most 1024 float4 values stays below 2**24 and float32 sums it exactly in
any order: the GPU's order cannot change the result.

The bank tile holds ``w * 2654435761 mod 2**32`` in word ``w``, and a lane's
output is the wrapping uint32 sum of the words it read, which does not
depend on the order of the loads. A digest hashes the little-endian bytes of
a float32 or uint32 buffer, so equal digests mean bit-identical buffers.
"""

from __future__ import annotations

import numpy

from .gpu_ceiling_source import (
    BANK_COLUMNS,
    BANK_LANES,
    BANK_LOADS_PER_TRIP,
    BANK_MAX_STRIDE,
    BANK_SHIFT_PERIOD,
    BANK_TILE_ROWS,
    BANK_WORD_MULTIPLIER,
    FMA_LANES,
    FMA_START_COLUMNS,
    WORD_BYTES,
    GpuCeilingError,
    bank_kernel_name,
    fma_kernel_name,
    require_count,
)
from .kernel_metal_geometry import ceil_div
from .kernel_metal_reference import MAX_SEED, array_sha256

FMA_STREAM = 2
COPY_STREAM = 3
REDUCE_STREAM = 4
STREAMS = (FMA_STREAM, COPY_STREAM, REDUCE_STREAM)
MAX_CASE_INDEX = 63

FMA_MULTIPLIER = numpy.float32((2**24 - 4173) * 2.0**-24)
FMA_ADDEND = numpy.float32(12561923 * 2.0**-35)
FMA_COEFFICIENTS = (float(FMA_MULTIPLIER), float(FMA_ADDEND))
# The CPU rebuild costs one NumPy step per FMA step of a chain.
MAX_FMA_STEPS = 1 << 16

MAX_STREAM_FLOAT4 = 1 << 26
MAX_REDUCE_VALUE = 1023
MAX_FLOAT4_PER_THREAD = 1024

MAX_BANK_TRIPS = 1 << 16
_LITTLE_WORDS = numpy.dtype("<u4")


def ceiling_generator(seed: int, stream: int, index: int) -> numpy.random.Generator:
    """Return the generator of one input, independent of every other input."""

    require_count(seed, "seed", 0, MAX_SEED)
    if type(stream) is not int or stream not in STREAMS:
        raise GpuCeilingError("stream must be one of the ceiling lab's streams")
    require_count(index, "case index", 0, MAX_CASE_INDEX)
    return numpy.random.default_rng([seed, stream, index])


def words_sha256(words: numpy.ndarray) -> str:
    """Hash a uint32 buffer through the same bytes as a float32 one."""

    if not isinstance(words, numpy.ndarray) or words.dtype != numpy.uint32:
        raise GpuCeilingError("only uint32 word buffers are hashed as words")
    little = words.astype(_LITTLE_WORDS, copy=False)
    return array_sha256(numpy.ascontiguousarray(little).view(numpy.float32))


def lane_tile(words: numpy.ndarray, threads: int) -> numpy.ndarray:
    """Spread 32 per-lane words over ``threads`` threads: thread i gets i % 32.

    Both the FMA and the bank kernels repeat every 32 threads, so 32 words
    rebuild a whole output buffer.
    """

    if (
        not isinstance(words, numpy.ndarray)
        or words.dtype != numpy.uint32
        or words.shape != (FMA_LANES,)
    ):
        raise GpuCeilingError("lane words must be 32 uint32 values")
    count = require_count(threads, "threads", 1, MAX_STREAM_FLOAT4)
    return words[numpy.arange(count) % FMA_LANES]


def fma_start_values(seed: int) -> numpy.ndarray:
    """Draw the 32 x 32 start table: one row per lane, one column per chain."""

    generator = ceiling_generator(seed, FMA_STREAM, 0)
    significands = generator.integers(
        2**23, 2**24, size=(FMA_LANES, FMA_START_COLUMNS), dtype=numpy.int64
    )
    return significands.astype(numpy.float32) * numpy.float32(2.0**-23)


def _steps(values: object) -> tuple[int, ...]:
    if type(values) is not tuple or not values:
        raise GpuCeilingError("steps must be a nonempty tuple")
    for value in values:
        require_count(value, "steps", 1, MAX_FMA_STEPS)
    if any(later <= earlier for earlier, later in zip(values, values[1:])):
        raise GpuCeilingError("steps must be strictly increasing")
    return values


def fma_chain_values(
    start: numpy.ndarray, steps: tuple[int, ...]
) -> tuple[numpy.ndarray, ...]:
    """Run every chain on the CPU and return the table after each step count.

    One pass serves every checkpoint, so a sweep of step counts costs the
    largest count once.
    """

    if (
        not isinstance(start, numpy.ndarray)
        or start.dtype != numpy.float32
        or start.shape != (FMA_LANES, FMA_START_COLUMNS)
    ):
        raise GpuCeilingError("start must be the 32 x 32 float32 start table")
    if not numpy.all((start >= 1.0) & (start < 2.0)):
        raise GpuCeilingError("start values must lie in [1, 2)")
    checkpoints = _steps(steps)
    multiplier, addend = (float(value) for value in (FMA_MULTIPLIER, FMA_ADDEND))
    values = start.copy()
    tables = []
    done = 0
    for target in checkpoints:
        for _ in range(target - done):
            exact = values.astype(numpy.float64) * multiplier + addend
            values = exact.astype(numpy.float32)
        done = target
        tables.append(values.copy())
    return tuple(tables)


def fma_lane_words(values: numpy.ndarray, chains: int) -> numpy.ndarray:
    """XOR the bit patterns of the first ``chains`` chains of each lane."""

    fma_kernel_name(chains)
    if (
        not isinstance(values, numpy.ndarray)
        or values.dtype != numpy.float32
        or values.shape != (FMA_LANES, FMA_START_COLUMNS)
    ):
        raise GpuCeilingError("values must be a 32 x 32 float32 table")
    bits = numpy.ascontiguousarray(values[:, :chains]).view(numpy.uint32)
    return numpy.bitwise_xor.reduce(bits, axis=1).astype(numpy.uint32)


def stream_threads(float4_count: int, per_thread: int) -> int:
    """Threads of a grid-stride walk: ``ceil(count / per_thread)``."""

    count = require_count(float4_count, "float4 count", 1, MAX_STREAM_FLOAT4)
    per = require_count(per_thread, "float4 per thread", 1, MAX_FLOAT4_PER_THREAD)
    return ceil_div(count, per)


def copy_source(seed: int, index: int, float4_count: int) -> numpy.ndarray:
    """Draw ``float4_count`` standard-normal float4 values as a flat float32 array."""

    count = require_count(float4_count, "float4 count", 1, MAX_STREAM_FLOAT4)
    generator = ceiling_generator(seed, COPY_STREAM, index)
    return generator.standard_normal(4 * count, dtype=numpy.float32)


def reduce_source(seed: int, index: int, float4_count: int) -> numpy.ndarray:
    """Draw whole numbers in [0, 1024) as float32, four per float4 value."""

    count = require_count(float4_count, "float4 count", 1, MAX_STREAM_FLOAT4)
    generator = ceiling_generator(seed, REDUCE_STREAM, index)
    values = generator.integers(
        0, MAX_REDUCE_VALUE + 1, size=4 * count, dtype=numpy.uint16
    )
    return values.astype(numpy.float32)


def read_reduce_partials(source: numpy.ndarray, threads: int) -> numpy.ndarray:
    """Sum float4 value ``e`` into partial ``e % threads``, exactly.

    The kernel's walk gives thread ``t`` the values ``t, t + threads, ...``.
    Whole numbers below 2**24 add exactly in float64, so the result is the
    GPU's in any order, and the sums run without a copy of the source.
    """

    if (
        not isinstance(source, numpy.ndarray)
        or source.dtype != numpy.float32
        or source.ndim != 1
        or source.size % 4
    ):
        raise GpuCeilingError("source must be a flat float32 array of float4 values")
    count = source.size // 4
    walkers = require_count(threads, "threads", 1, count)
    if ceil_div(count, walkers) > MAX_FLOAT4_PER_THREAD:
        raise GpuCeilingError(
            f"a thread would sum more than {MAX_FLOAT4_PER_THREAD} float4 values, "
            "the most that stay exact"
        )
    rows, rest = divmod(count, walkers)
    values = source.reshape(count, 4)
    sums = numpy.zeros(walkers, dtype=numpy.float64)
    if rows:
        full = values[: rows * walkers].reshape(rows, walkers, 4)
        sums += full.sum(axis=(0, 2), dtype=numpy.float64)
    if rest:
        sums[:rest] += values[rows * walkers :].sum(axis=1, dtype=numpy.float64)
    return sums.astype(numpy.float32)


def bank_tile(row_words: int) -> numpy.ndarray:
    """The tile's words as the kernel writes them: ``w * 2654435761 mod 2**32``."""

    bank_kernel_name(row_words)
    words = numpy.arange(BANK_TILE_ROWS * row_words, dtype=numpy.uint64)
    return (words * BANK_WORD_MULTIPLIER % 2**32).astype(numpy.uint32)


def bank_first_words(stride: int, row_words: int) -> numpy.ndarray:
    """Each lane's first word: logical ``lane * stride`` in a 32-column tile."""

    require_count(stride, "stride", 1, BANK_MAX_STRIDE)
    bank_kernel_name(row_words)
    logical = numpy.arange(BANK_LANES, dtype=numpy.int64) * stride
    return logical // BANK_COLUMNS * row_words + logical % BANK_COLUMNS


def bank_lane_addresses(stride: int, row_words: int) -> tuple[int, ...]:
    """Byte addresses of one load across the 32 lanes, for the bank model.

    Every trip moves all lanes by whole rows, so this one load has the bank
    pattern of every load.
    """

    return tuple(int(word) * WORD_BYTES for word in bank_first_words(stride, row_words))


def bank_lane_sums(stride: int, row_words: int, trips: int) -> numpy.ndarray:
    """Rebuild the wrapping uint32 sum that each of the 32 lanes returns.

    The row shift repeats every 64 trips, so one period of loads, counted
    whole and then in part, covers any trip count.
    """

    count = require_count(trips, "trips", 1, MAX_BANK_TRIPS)
    tile = bank_tile(row_words).astype(numpy.uint64)
    first = bank_first_words(stride, row_words)
    rows = numpy.arange(BANK_SHIFT_PERIOD)[:, None] + numpy.arange(BANK_LOADS_PER_TRIP)
    per_shift = tile[first[:, None, None] + rows[None] * row_words].sum(axis=2)
    cycles, rest = divmod(count, BANK_SHIFT_PERIOD)
    sums = per_shift.sum(axis=1) * numpy.uint64(cycles)
    sums += per_shift[:, :rest].sum(axis=1, dtype=numpy.uint64)
    return (sums % 2**32).astype(numpy.uint32)
