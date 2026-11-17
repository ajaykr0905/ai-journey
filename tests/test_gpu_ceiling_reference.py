from fractions import Fraction

import numpy
import pytest

from ai_journey.gpu_ceiling_reference import (
    FMA_ADDEND,
    FMA_COEFFICIENTS,
    FMA_MULTIPLIER,
    MAX_FLOAT4_PER_THREAD,
    MAX_FMA_STEPS,
    bank_first_words,
    bank_lane_addresses,
    bank_lane_sums,
    bank_tile,
    ceiling_generator,
    copy_source,
    fma_chain_values,
    fma_lane_words,
    fma_start_values,
    lane_tile,
    read_reduce_partials,
    reduce_source,
    stream_threads,
    words_sha256,
)
from ai_journey.gpu_ceiling_source import (
    BANK_LOADS_PER_TRIP,
    BANK_SHIFT_PERIOD,
    BANK_TILE_ROWS,
    FMA_CHAIN_COUNTS,
    FMA_STEPS_PER_TRIP,
    GpuCeilingError,
)
from ai_journey.kernel_metal_reference import array_sha256

A = Fraction(float(FMA_MULTIPLIER))
B = Fraction(float(FMA_ADDEND))


def exact_fma(x):
    """Round x * a + b once, to nearest even, at float32 precision in [1, 2)."""

    exact = Fraction(float(x)) * A + B
    assert 1 <= exact < 2
    return numpy.float32(round(exact * 2**23) / 2**23)


def test_the_coefficients_are_exact_float32_values_with_full_significands():
    assert A == Fraction(2**24 - 4173, 2**24)
    assert B == Fraction(12561923, 2**35)
    assert 2**23 <= A * 2**24 < 2**24 and 2**23 <= B * 2**35 < 2**24
    assert FMA_COEFFICIENTS == (float(A), float(B))
    # The fixed point b / (1 - a) lies inside [1, 2).
    assert 1.46 < B / (1 - A) < 1.48


def test_start_values_fill_one_binade_and_rebuild_from_the_seed():
    start = fma_start_values(74)

    assert start.shape == (32, 32) and start.dtype == numpy.float32
    assert start.min() >= 1.0 and start.max() < 2.0
    assert numpy.array_equal(start, fma_start_values(74))
    assert not numpy.array_equal(start, fma_start_values(75))
    assert len(numpy.unique(start)) == start.size


def test_each_cpu_step_is_one_correctly_rounded_fused_multiply_add():
    start = fma_start_values(74)
    (after_one, after_five) = fma_chain_values(start, (1, 5))

    expected = start.copy()
    for step in range(5):
        expected = numpy.vectorize(exact_fma, otypes=[numpy.float32])(expected)
        if step == 0:
            assert numpy.array_equal(after_one, expected)
    assert numpy.array_equal(after_five, expected)


def test_an_unfused_multiply_then_add_changes_most_chains():
    start = fma_start_values(74)
    (fused,) = fma_chain_values(start, (256,))

    unfused = start.copy()
    for _ in range(256):
        unfused = (unfused * FMA_MULTIPLIER).astype(numpy.float32) + FMA_ADDEND
    assert (unfused != fused).mean() > 0.9


def test_chains_stay_in_the_binade_and_keep_moving_across_the_sweep():
    start = fma_start_values(74)
    tables = fma_chain_values(start, (32, 2048, 8192))

    for table in tables:
        assert table.min() >= 1.0 and table.max() < 2.0
    assert (tables[1] != tables[2]).mean() > 0.9
    with pytest.raises(GpuCeilingError, match="strictly increasing"):
        fma_chain_values(start, (64, 64))
    with pytest.raises(GpuCeilingError, match=r"\[1, 2\)"):
        fma_chain_values(start * 2, (1,))


def fused_step(values):
    exact = values.astype(numpy.float64) * float(FMA_MULTIPLIER) + float(FMA_ADDEND)
    return exact.astype(numpy.float32)


@pytest.mark.parametrize("seed", [0, 74, 2**32 - 1])
def test_at_the_step_bound_one_trip_more_still_changes_most_lane_words(seed):
    # A timed output proves that its trips ran only while the chains move.
    before, last = fma_chain_values(
        fma_start_values(seed), (MAX_FMA_STEPS - FMA_STEPS_PER_TRIP, MAX_FMA_STEPS)
    )

    assert (before != last).mean() > 0.9
    for chains in FMA_CHAIN_COUNTS:
        words = fma_lane_words(last, chains) != fma_lane_words(before, chains)
        assert words.mean() > 0.9, chains


def test_past_twice_the_step_bound_rounding_has_stopped_every_chain():
    # x stops once it is within 1/4173 of b / (1 - a); from the farthest start
    # that takes about 31000 steps, so more steps could not change an output.
    (values,) = fma_chain_values(fma_start_values(74), (MAX_FMA_STEPS,))
    for _ in range(MAX_FMA_STEPS):
        values = fused_step(values)

    assert numpy.array_equal(fused_step(values), values)
    with pytest.raises(GpuCeilingError, match="steps must be in"):
        fma_chain_values(fma_start_values(74), (MAX_FMA_STEPS + FMA_STEPS_PER_TRIP,))


def test_lane_words_xor_every_chain_and_tile_over_threads():
    start = fma_start_values(74)
    bits = start.view(numpy.uint32)

    for chains in (1, 2, 4, 8, 16, 32):
        words = fma_lane_words(start, chains)
        assert words.dtype == numpy.uint32 and words.shape == (32,)
        expected = bits[:, 0].copy()
        for column in range(1, chains):
            expected ^= bits[:, column]
        assert numpy.array_equal(words, expected), chains
    changed = start.copy()
    changed[3, 7] = numpy.nextafter(changed[3, 7], numpy.float32(2))
    assert fma_lane_words(changed, 8)[3] != fma_lane_words(start, 8)[3]
    assert fma_lane_words(changed, 4)[3] == fma_lane_words(start, 4)[3]
    tiled = lane_tile(fma_lane_words(start, 4), 1000)
    assert tiled.shape == (1000,) and tiled[999] == tiled[999 % 32]


def test_word_digests_match_the_float32_bytes_of_the_same_buffer():
    words = numpy.arange(5, dtype=numpy.uint32) * numpy.uint32(0x9E3779B1)

    assert words_sha256(words) == array_sha256(words.view(numpy.float32))
    with pytest.raises(GpuCeilingError, match="uint32"):
        words_sha256(words.astype(numpy.int64))


def test_stream_sources_are_seeded_and_reduce_sources_are_small_integers():
    copied = copy_source(74, 0, 4099)
    summed = reduce_source(74, 0, 4099)

    assert copied.shape == summed.shape == (4 * 4099,)
    assert numpy.array_equal(copied, copy_source(74, 0, 4099))
    assert not numpy.array_equal(copied, copy_source(74, 1, 4099))
    assert summed.min() >= 0 and summed.max() <= 1023
    assert numpy.array_equal(summed, numpy.round(summed))
    assert stream_threads(4099, 16) == 257
    with pytest.raises(GpuCeilingError, match="streams"):
        ceiling_generator(74, 0, 0)


@pytest.mark.parametrize(("count", "threads"), [(4099, 257), (4096, 256), (7, 7)])
def test_read_reduce_partials_are_exact_integer_sums_of_the_walk(count, threads):
    source = reduce_source(74, 2, count)
    values = source.reshape(count, 4).astype(numpy.int64)

    partials = read_reduce_partials(source, threads)

    expected = [int(values[t::threads].sum()) for t in range(threads)]
    assert partials.dtype == numpy.float32
    assert partials.astype(numpy.int64).tolist() == expected


def test_read_reduce_refuses_a_walk_that_could_round():
    source = reduce_source(74, 0, MAX_FLOAT4_PER_THREAD + 1)

    with pytest.raises(GpuCeilingError, match="stay exact"):
        read_reduce_partials(source, 1)
    assert read_reduce_partials(source, 2).shape == (2,)


@pytest.mark.parametrize("row_words", [32, 33])
def test_bank_lane_sums_match_a_direct_walk_of_the_tile(row_words):
    tile = [int(word) for word in bank_tile(row_words)]
    assert len(tile) == BANK_TILE_ROWS * row_words
    assert tile[5] == 5 * 2654435761 % 2**32

    for stride, trips in ((1, 3), (7, 70), (32, 129)):
        first = bank_first_words(stride, row_words)
        expected = []
        for lane in range(32):
            total = 0
            for trip in range(trips):
                for load in range(BANK_LOADS_PER_TRIP):
                    row = trip % BANK_SHIFT_PERIOD + load
                    total += tile[int(first[lane]) + row * row_words]
            expected.append(total % 2**32)
        sums = bank_lane_sums(stride, row_words, trips)
        assert sums.dtype == numpy.uint32
        assert sums.tolist() == expected, (stride, trips)


def test_padding_moves_each_logical_row_by_one_word():
    unpadded = bank_lane_addresses(32, 32)
    padded = bank_lane_addresses(32, 33)

    assert unpadded == tuple(4 * 32 * lane for lane in range(32))
    assert padded == tuple(4 * 33 * lane for lane in range(32))
    assert bank_lane_addresses(1, 33) == tuple(4 * lane for lane in range(32))
    assert bank_lane_addresses(2, 33)[16] == 4 * 33
    with pytest.raises(GpuCeilingError, match="stride must be in"):
        bank_lane_addresses(33, 32)
