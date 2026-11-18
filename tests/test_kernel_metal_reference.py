import math
from fractions import Fraction

import numpy
import pytest

from ai_journey.kernel_metal_geometry import (
    MAX_THREADGROUP_THREADS,
    KernelMetalError,
    vector_launch,
)
from ai_journey.kernel_metal_reference import (
    FLOAT32_SENTINEL_BITS,
    GRAY_WEIGHTS,
    MAX_IMAGE_PIXELS,
    MAX_LEVEL_DIFFERENCE,
    MAX_RECORDED_MISMATCHES,
    MAX_VECTOR_LENGTH,
    UINT8_SENTINEL,
    LevelComparison,
    all_rgb_triples,
    array_sha256,
    compare_levels,
    float32_sentinels,
    rgb_to_gray_reference,
    uint8_sentinels,
    uniform_rgb_image,
    vector_add_inputs,
    vector_add_reference,
)

# Digests recorded with NumPy 1.26.4. A change means old reports can no
# longer be rebuilt from their seed, which the verifier must reject.
RECORDED = {
    "vector_add_a": "4451a455157d5d8db9f6dbe93fa1fff29337a55abfedf714e997e44ef35da1ba",
    "vector_add_b": "d1a6fef9d5c134013ee3332dcbbea445ef71df5de015559bcb916a03cc13deb3",
    "vector_add_sum": "b56ad36e27e6aa0d43e4fea6cfa07dde9dc5f52a6045ab5b0786e485623b42d2",
    "image": "2c2b39a98d773e94e2af75d529a1d042f3e7cf439621a4acc69b9be8d369b11d",
    "gray": "fd17eeaccc763bbf357490f3b0712eb4748da7fcc8f9c73aed9c762dd233cb9c",
    "all_triples": "95eeb80877c99cdcb38755b9bb5ed29066bf70e870ea6eff9ee30285bd4cd5b7",
    "all_triples_gray": "fd765e780c79467c5a60363988d11528e0e76a30efe4d37517897ca0fa08f9c1",
}


def gray_of(*rgb):
    pixel = numpy.array(rgb, dtype=numpy.uint8).reshape(1, 1, 3)
    return int(rgb_to_gray_reference(pixel)[0, 0])


def test_seeded_inputs_and_references_reproduce_their_recorded_digests():
    first, second = vector_add_inputs(73, 0, 1000)
    image = uniform_rgb_image(73, 0, 17, 33)

    assert (first.dtype, first.shape, second.shape) == (numpy.float32, (1000,), (1000,))
    assert array_sha256(first) == RECORDED["vector_add_a"]
    assert array_sha256(second) == RECORDED["vector_add_b"]
    assert (
        array_sha256(vector_add_reference(first, second)) == RECORDED["vector_add_sum"]
    )
    assert (image.dtype, image.shape) == (numpy.uint8, (17, 33, 3))
    assert array_sha256(image) == RECORDED["image"]
    assert array_sha256(rgb_to_gray_reference(image)) == RECORDED["gray"]


def test_each_case_draws_from_its_own_stream():
    base = array_sha256(vector_add_inputs(73, 0, 64)[0])

    assert array_sha256(vector_add_inputs(73, 1, 64)[0]) != base
    assert array_sha256(vector_add_inputs(74, 0, 64)[0]) != base
    assert array_sha256(uniform_rgb_image(73, 0, 4, 4)) != array_sha256(
        uniform_rgb_image(73, 1, 4, 4)
    )


def test_grayscale_rounds_half_up_where_banker_rounding_would_go_down():
    # 1140 * 250 + 5000 is 290000: the exact sum 28.5 rounds up to 29.
    assert gray_of(0, 0, 250) == 29
    assert numpy.round(28.5) == 28
    assert [gray_of(0, 0, 0), gray_of(255, 255, 255)] == [0, 255]
    assert [gray_of(255, 0, 0), gray_of(0, 255, 0), gray_of(0, 0, 255)] == [76, 150, 29]


def test_the_all_triples_image_covers_the_whole_input_domain_once():
    image = all_rgb_triples()
    channels = [image.reshape(-1, 3)[:, axis].astype(numpy.int32) for axis in range(3)]
    codes = (channels[0] << 16) | (channels[1] << 8) | channels[2]

    assert image.shape == (4096, 4096, 3)
    assert numpy.array_equal(codes, numpy.arange(1 << 24))
    assert array_sha256(image) == RECORDED["all_triples"]
    assert array_sha256(rgb_to_gray_reference(image)) == RECORDED["all_triples_gray"]


def test_every_exact_tie_rounds_up_as_in_the_cuda_half():
    image = all_rgb_triples()
    pixels = image.reshape(-1, 3)
    gray = rgb_to_gray_reference(image).reshape(-1)
    # Round half up as a quotient and a carry, apart from the reference's
    # add-then-divide form: the weighted sum times 10000 is an integer.
    total = sum(
        weight * pixels[:, axis].astype(numpy.int32)
        for axis, weight in enumerate((2989, 5870, 1140))
    )
    quotient, remainder = numpy.divmod(total, 10000)

    assert numpy.array_equal(gray, quotient + (remainder >= 5000))
    assert MAX_LEVEL_DIFFERENCE == 0
    ties = numpy.flatnonzero(remainder == 5000)
    assert ties.size == 1703
    decimals = [Fraction(f"{weight:.4f}") for weight in GRAY_WEIGHTS]
    for index in ties.tolist():
        exact = sum(w * int(v) for w, v in zip(decimals, pixels[index]))
        assert exact - math.floor(exact) == Fraction(1, 2)
        assert int(gray[index]) == math.floor(exact) + 1
    # The float32 form of the rule, which this lab used before it took the
    # integer form, rounds 19 of the ties down.
    red, green, blue = (numpy.float32(weight) for weight in GRAY_WEIGHTS)
    tie = pixels[ties].astype(numpy.float32)
    level = (tie[:, 0] * red + tie[:, 1] * green) + tie[:, 2] * blue
    rounded = numpy.floor(level + numpy.float32(0.5))
    assert int(numpy.count_nonzero(rounded < gray[ties])) == 19


def test_comparisons_record_every_mismatch_needed_to_rebuild_the_output():
    reference = rgb_to_gray_reference(uniform_rgb_image(73, 2, 8, 9))
    output = reference.copy()
    output[0, 3] = (int(reference[0, 3]) + 1) % 256
    output[5, 8] = abs(int(reference[5, 8]) - 1)

    comparison = compare_levels(output, reference)

    assert comparison.mismatches == (
        (3, int(output[0, 3])),
        (53, int(output[5, 8])),
    )
    assert comparison.max_abs_difference == max(
        abs(int(output[0, 3]) - int(reference[0, 3])),
        abs(int(output[5, 8]) - int(reference[5, 8])),
    )
    rebuilt = reference.reshape(-1).copy()
    for index, level in comparison.mismatches:
        rebuilt[index] = level
    assert array_sha256(rebuilt) == array_sha256(output.reshape(-1))
    assert comparison.to_mapping()["mismatched_pixels"] == 2
    equal = compare_levels(reference, reference)
    assert equal.to_mapping() == {
        "mismatches": [],
        "mismatched_pixels": 0,
        "max_abs_difference_levels": 0,
    }


def test_comparisons_refuse_more_mismatches_than_a_report_can_hold():
    reference = numpy.zeros(MAX_RECORDED_MISMATCHES + 1, dtype=numpy.uint8)

    with pytest.raises(KernelMetalError, match="at most 1024 can be recorded"):
        compare_levels(reference + 1, reference)
    with pytest.raises(KernelMetalError, match="shapes differ"):
        compare_levels(reference[:-1], reference)
    with pytest.raises(KernelMetalError, match="uint8"):
        compare_levels(reference.astype(numpy.int16), reference)
    for mismatches, difference, message in (
        (((2, 1), (1, 1)), 1, "strictly increasing"),
        (((9, 1),), 1, "mismatch index"),
        (((1, 256),), 1, "mismatch level"),
        ((), 1, "nonzero difference"),
        ([(1, 1)], 1, "at most 1024"),
    ):
        with pytest.raises(KernelMetalError, match=message):
            LevelComparison(
                pixels=8, mismatches=mismatches, max_abs_difference=difference
            )


def test_sentinels_have_fixed_bits_and_digests_cover_exact_bytes():
    nans = float32_sentinels(3)

    assert nans.view(numpy.uint32).tolist() == [FLOAT32_SENTINEL_BITS] * 3
    assert uint8_sentinels(2).tolist() == [UINT8_SENTINEL] * 2
    assert array_sha256(float32_sentinels(0)) == array_sha256(
        numpy.zeros(0, dtype=numpy.uint8)
    )
    big_endian = numpy.arange(4, dtype=">f4")
    assert array_sha256(big_endian.astype(numpy.float32)) == array_sha256(
        numpy.arange(4, dtype=numpy.float32)
    )
    with pytest.raises(KernelMetalError, match="float32 and uint8"):
        array_sha256(numpy.arange(4, dtype=numpy.float64))


def test_sentinels_fill_the_longest_vector_launch_of_any_group_size():
    # A group size that does not divide the length adds up to group - 1
    # padding threads, so the guarded buffer can pass MAX_VECTOR_LENGTH.
    longest = max(
        vector_launch(MAX_VECTOR_LENGTH, group).launched_threads
        for group in range(1, MAX_THREADGROUP_THREADS + 1)
    )

    assert longest > MAX_VECTOR_LENGTH
    assert float32_sentinels(longest).size == longest


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: vector_add_inputs(73, 0, 0), KernelMetalError, "length"),
        (lambda: vector_add_inputs(-1, 0, 4), KernelMetalError, "seed"),
        (lambda: vector_add_inputs(73, 64, 4), KernelMetalError, "case index"),
        (lambda: vector_add_inputs(73, 0, 4.0), TypeError, "length"),
        (lambda: uniform_rgb_image(73, 0, 0, 4), KernelMetalError, "height"),
        (lambda: uniform_rgb_image(73, 0, 8192, 8192), KernelMetalError, "pixels"),
        (
            lambda: vector_add_reference(
                numpy.zeros(3, numpy.float32), numpy.zeros(4, numpy.float32)
            ),
            KernelMetalError,
            "same length",
        ),
        (
            lambda: vector_add_reference(
                numpy.zeros(3, numpy.float64), numpy.zeros(3, numpy.float32)
            ),
            KernelMetalError,
            "float32",
        ),
        (
            lambda: rgb_to_gray_reference(numpy.zeros((2, 2, 4), numpy.uint8)),
            KernelMetalError,
            "height, width, 3",
        ),
    ],
)
def test_invalid_inputs_are_rejected(call, error, message):
    with pytest.raises(error, match=message):
        call()
    assert MAX_IMAGE_PIXELS == 4096 * 4096
