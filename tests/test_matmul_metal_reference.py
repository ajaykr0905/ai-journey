import itertools
from hashlib import sha256

import numpy
import pytest

from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.matmul_metal_mapping import MatmulMetalError, matmul_shape
from ai_journey.matmul_metal_reference import (
    BENCHMARK_STREAM,
    CORRECTNESS_STREAM,
    error_bound,
    float64_reference,
    gamma,
    matmul_inputs,
    matrix_sha256,
    max_relative_error,
)


def sequential_python_sum(left, right):
    """The reference rule in plain Python floats, which are IEEE doubles."""

    rows, inner = left.shape
    columns = right.shape[1]
    result = numpy.zeros((rows, columns))
    for row, column in itertools.product(range(rows), range(columns)):
        total = 0.0
        for step in range(inner):
            total += float(left[row, step]) * float(right[step, column])
        result[row, column] = total
    return result


def test_inputs_are_seeded_standard_normal_float32_per_case():
    shape = matmul_shape(17, 33, 9)

    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 3, shape)
    again = matmul_inputs(74, CORRECTNESS_STREAM, 3, shape)

    assert (left.shape, right.shape) == ((17, 9), (9, 33))
    assert left.dtype == right.dtype == numpy.float32
    assert numpy.array_equal(left, again[0]) and numpy.array_equal(right, again[1])
    generator = numpy.random.default_rng([74, CORRECTNESS_STREAM, 3])
    expected = generator.standard_normal((17, 9), dtype=numpy.float32)
    assert numpy.array_equal(left, expected)
    for other in (
        matmul_inputs(74, CORRECTNESS_STREAM, 4, shape),
        matmul_inputs(74, BENCHMARK_STREAM, 3, shape),
        matmul_inputs(75, CORRECTNESS_STREAM, 3, shape),
    ):
        assert not numpy.array_equal(left, other[0])
    for bad in ((-1, 0, 0), (74, 2, 0), (74, 0, 64)):
        with pytest.raises(KernelMetalError, match="must be in"):
            matmul_inputs(*bad, shape)


def test_the_reference_adds_exact_products_in_k_order():
    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 0, matmul_shape(5, 7, 33))

    reference, magnitude = float64_reference(left, right)

    expected = sequential_python_sum(left, right)
    assert reference.dtype == magnitude.dtype == numpy.float64
    assert numpy.array_equal(reference, expected)
    assert numpy.array_equal(
        magnitude, sequential_python_sum(numpy.abs(left), numpy.abs(right))
    )
    assert (magnitude >= numpy.abs(reference)).all()


def test_the_bound_grows_with_k_and_states_both_roundings():
    bounds = [error_bound(k) for k in (1, 64, 1024, 4096, 8192)]

    assert bounds == sorted(bounds)
    assert error_bound(1) == 2.0**-24 / (1 - 2.0**-24) + 2.0**-53 / (1 - 2.0**-53)
    assert gamma(4096, 2.0**-24) == 4096 * 2.0**-24 / (1 - 4096 * 2.0**-24)
    assert 2.442e-4 < error_bound(4096) < 2.443e-4
    for bad in (0, 8193):
        with pytest.raises(KernelMetalError, match="k must be in"):
            error_bound(bad)


def test_relative_error_is_scaled_by_the_absolute_products():
    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 1, matmul_shape(31, 47, 129))
    reference, magnitude = float64_reference(left, right)
    product = left @ right

    error = max_relative_error(product, reference, magnitude)

    assert 0 < error <= error_bound(129)
    assert max_relative_error(reference.astype(numpy.float32), reference, magnitude) < (
        2.0**-24
    )
    wrong = product.copy()
    wrong[3, 5] += numpy.float32(4 * error_bound(129) * magnitude[3, 5])
    assert max_relative_error(wrong, reference, magnitude) > error_bound(129)


def test_zero_scales_count_matches_as_zero_and_mismatches_as_infinite():
    reference = numpy.zeros((2, 2))
    magnitude = numpy.zeros((2, 2))
    output = numpy.zeros((2, 2), dtype=numpy.float32)

    assert max_relative_error(output, reference, magnitude) == 0.0
    output[1, 0] = 1e-30
    assert max_relative_error(output, reference, magnitude) == float("inf")
    output[1, 0] = numpy.nan
    with pytest.raises(MatmulMetalError, match="NaN or an infinity"):
        max_relative_error(output, reference, magnitude)
    with pytest.raises(MatmulMetalError, match="reference shape"):
        max_relative_error(output[:1], reference, magnitude)
    with pytest.raises(MatmulMetalError, match="conformable 2D float32"):
        float64_reference(numpy.ones((2, 3)), numpy.ones((3, 2)))


def test_digests_hash_little_endian_bytes_and_tell_bits_apart():
    values = numpy.array([[0.0, -0.0], [1.5, 2.0]])

    assert matrix_sha256(values) == sha256(values.astype("<f8").tobytes()).hexdigest()
    assert matrix_sha256(values.astype(">f8")) == matrix_sha256(values)
    single = values.astype(numpy.float32)
    assert matrix_sha256(single) == sha256(single.astype("<f4").tobytes()).hexdigest()
    flipped = values.copy()
    flipped[0, 1] = 0.0
    assert matrix_sha256(flipped) != matrix_sha256(values)
    with pytest.raises(MatmulMetalError, match="only float32 and float64"):
        matrix_sha256(values.astype(numpy.float16))
