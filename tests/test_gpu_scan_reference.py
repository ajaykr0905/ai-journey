import math
from hashlib import sha256

import numpy
import pytest

from ai_journey.gpu_scan_reference import (
    BENCHMARK_STREAM,
    CORRECTNESS_STREAM,
    INTEGERS,
    MAX_EXACT_LENGTH,
    NORMAL,
    U32,
    U64,
    blelloch_replay,
    error_bound,
    exact_prefix,
    float64_reference,
    float64_sha256,
    gamma,
    hillis_steele_replay,
    max_relative_error,
    scan_inputs,
)
from ai_journey.gpu_scan_source import (
    BLOCK,
    S0_HILLIS_STEELE,
    S1_BLELLOCH,
    S2_SIMD,
    GpuScanError,
)

F32 = numpy.float32


def naive_hillis_steele_block(values):
    """One block, one element at a time: the S0 text read line by line."""

    data = [F32(0)] * BLOCK
    data[: len(values)] = [F32(value) for value in values]
    offset = 1
    while offset < BLOCK:
        data = [
            F32(data[t - offset] + data[t]) if t >= offset else data[t]
            for t in range(BLOCK)
        ]
        offset *= 2
    return data


def naive_blelloch_block(values):
    tree = [F32(0)] * BLOCK
    tree[: len(values)] = [F32(value) for value in values]
    stride, active = 1, BLOCK // 2
    while active:
        for t in range(active):
            right = (2 * t + 2) * stride - 1
            tree[right] = F32(tree[right - stride] + tree[right])
        stride, active = stride * 2, active // 2
    total, tree[BLOCK - 1] = tree[BLOCK - 1], F32(0)
    active = 1
    while active <= BLOCK // 2:
        stride //= 2
        for t in range(active):
            right = (2 * t + 2) * stride - 1
            carry = tree[right - stride]
            tree[right - stride] = tree[right]
            tree[right] = F32(carry + tree[right])
        active *= 2
    return [*tree[1:], total]


def naive_levels(values, block_scan):
    blocks = [values[start : start + BLOCK] for start in range(0, len(values), BLOCK)]
    scanned = [block_scan(block) for block in blocks]
    prefix = [value for block in scanned for value in block][: len(values)]
    if len(blocks) == 1:
        return prefix
    offsets = naive_levels([block[-1] for block in scanned], block_scan)
    return [
        F32(value + offsets[i // BLOCK - 1]) if i >= BLOCK else value
        for i, value in enumerate(prefix)
    ]


def bits(values):
    return numpy.asarray(values, dtype=numpy.float32).view(numpy.uint32)


@pytest.mark.parametrize("length", [1, 5, 33, 1023, 1024, 1025, 3000])
def test_the_vectorized_replays_equal_the_kernel_texts_read_line_by_line(length):
    values = scan_inputs(78, CORRECTNESS_STREAM, 2, length, NORMAL)

    hillis = naive_levels(list(values), naive_hillis_steele_block)
    blelloch = naive_levels(list(values), naive_blelloch_block)

    assert numpy.array_equal(bits(hillis_steele_replay(values)), bits(hillis))
    assert numpy.array_equal(bits(blelloch_replay(values)), bits(blelloch))


def test_the_two_replays_add_in_different_orders():
    values = scan_inputs(78, CORRECTNESS_STREAM, 3, 4097, NORMAL)

    hillis, blelloch = hillis_steele_replay(values), blelloch_replay(values)

    assert not numpy.array_equal(bits(hillis), bits(blelloch))
    assert numpy.count_nonzero(bits(hillis) != bits(blelloch)) > 100


@pytest.mark.parametrize("length", [1000, 1048577])
def test_each_replay_stays_within_the_bound_of_its_order(length):
    values = scan_inputs(78, CORRECTNESS_STREAM, 4, length, NORMAL)
    reference, scale = float64_reference(values)
    rounded = reference.astype(numpy.float32)

    for name, output in (
        (S0_HILLIS_STEELE, hillis_steele_replay(values)),
        (S1_BLELLOCH, blelloch_replay(values)),
        (S2_SIMD, rounded),
    ):
        error = max_relative_error(output, reference, scale)
        assert 0.0 < error <= error_bound(name, length) / 8, name


def test_small_integers_give_exact_sums_in_every_order():
    values = scan_inputs(78, CORRECTNESS_STREAM, 5, 70001, INTEGERS)
    exact = exact_prefix(values)

    assert set(numpy.unique(values)) <= set(range(-4, 5))
    assert numpy.array_equal(bits(hillis_steele_replay(values)), bits(exact))
    assert numpy.array_equal(bits(blelloch_replay(values)), bits(exact))
    sequential = numpy.cumsum(values, dtype=numpy.float32)
    assert numpy.array_equal(bits(sequential), bits(exact))
    with pytest.raises(GpuScanError, match="small integers"):
        exact_prefix(values + numpy.float32(0.5))


def test_every_case_draws_from_its_own_stream():
    first = scan_inputs(78, CORRECTNESS_STREAM, 0, 4096, NORMAL)

    assert numpy.array_equal(
        first, scan_inputs(78, CORRECTNESS_STREAM, 0, 4096, NORMAL)
    )
    assert first.dtype == numpy.float32 and first.shape == (4096,)
    for other in (
        scan_inputs(79, CORRECTNESS_STREAM, 0, 4096, NORMAL),
        scan_inputs(78, BENCHMARK_STREAM, 0, 4096, NORMAL),
        scan_inputs(78, CORRECTNESS_STREAM, 1, 4096, NORMAL),
    ):
        assert not numpy.array_equal(first, other)
    assert numpy.array_equal(
        first[:100], numpy.random.default_rng([78, 0, 0]).standard_normal(100, F32)
    )


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ((78, 2, 0, 8, NORMAL), GpuScanError, "stream"),
        ((78, 0, 64, 8, NORMAL), GpuScanError, "case index"),
        ((-1, 0, 0, 8, NORMAL), GpuScanError, "seed"),
        ((78, 0, 0, 8, "uniform"), GpuScanError, "pattern must be one of"),
        ((78, 0, 0, MAX_EXACT_LENGTH + 1, INTEGERS), GpuScanError, "at most"),
        ((78, 0, 0, 0, NORMAL), GpuScanError, "length"),
        ((78, 0, True, 8, NORMAL), TypeError, "integer"),
    ],
)
def test_inputs_out_of_their_bounds_are_refused(arguments, error, message):
    with pytest.raises(error, match=message):
        scan_inputs(*arguments)


def test_the_bound_follows_the_declared_depth_and_the_float64_reference():
    wide = gamma(1024, U64)

    assert error_bound(S0_HILLIS_STEELE, 1024) == (gamma(10, U32) + wide) / (1 - wide)
    assert error_bound(S1_BLELLOCH, 1 << 24) == (
        gamma(59, U32) + gamma(1 << 24, U64)
    ) / (1 - gamma(1 << 24, U64))
    assert error_bound(S2_SIMD, 1 << 20) > error_bound(S1_BLELLOCH, 1 << 20)
    assert gamma(10, U32) == 10 * U32 / (1 - 10 * U32)
    with pytest.raises(GpuScanError, match="below 1"):
        gamma(1 << 24, U32)


def test_the_error_counts_equal_elements_as_zero_and_never_hides_a_nan():
    reference = numpy.array([0.0, 1.0, 3.0])
    scale = numpy.array([0.0, 1.0, 3.0])

    assert max_relative_error(numpy.array([0, 1, 3], F32), reference, scale) == 0.0
    assert max_relative_error(numpy.array([0, 1, 2], F32), reference, scale) == 1 / 3
    assert max_relative_error(numpy.array([1, 1, 3], F32), reference, scale) == math.inf
    nan = numpy.array([0, numpy.nan, 3], F32)
    assert max_relative_error(nan, reference, scale) == math.inf
    with pytest.raises(GpuScanError, match="float32 vector"):
        max_relative_error(numpy.zeros(3), reference, scale)


def test_the_float64_digest_hashes_little_endian_bytes():
    reference, _ = float64_reference(numpy.array([1, 2], F32))

    assert (
        float64_sha256(reference)
        == sha256(numpy.array([1.0, 3.0], dtype="<f8").tobytes()).hexdigest()
    )
    assert float64_sha256(reference.astype(">f8")) == float64_sha256(reference)
    with pytest.raises(GpuScanError, match="float64"):
        float64_sha256(numpy.zeros(2, F32))
