import numpy
import pytest

from ai_journey.kernel_metal_geometry import (
    MAX_THREADGROUP_THREADS,
    KernelMetalError,
    vector_launch,
)
from ai_journey.kernel_metal_reference import (
    FLOAT32_SENTINEL_BITS,
    MAX_IMAGE_PIXELS,
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
    "all_triples_gray": "73b98ca1d9177e3bcea9816a73f18bcd9f4080ad50a6af8c28408ff6726bbaa4",
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
    # 0.1140 * 250 is 28.5 in decimal, and the float32 product rounds to 28.5.
    assert gray_of(0, 0, 250) == 29
    assert numpy.round(28.5) == 28
    assert [gray_of(0, 0, 0), gray_of(255, 255, 255)] == [0, 255]
    assert [gray_of(255, 0, 0), gray_of(0, 255, 0), gray_of(0, 0, 255)] == [76, 150, 29]


def test_the_all_triples_image_covers_the_whole_input_domain_once():
    image = all_rgb_triples()
    flat = image.reshape(-1, 3).astype(numpy.int64)
    codes = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]

    assert image.shape == (4096, 4096, 3)
    assert numpy.array_equal(codes, numpy.arange(1 << 24))
    assert array_sha256(image) == RECORDED["all_triples"]
    gray = rgb_to_gray_reference(image)
    assert array_sha256(gray) == RECORDED["all_triples_gray"]
    # Exact decimal rounding as integers: floor((2989 R + 5870 G + 1140 B
    # + 5000) / 10000). Float32 rounding drops 19 triples one level below it.
    exact = (2989 * flat[:, 0] + 5870 * flat[:, 1] + 1140 * flat[:, 2] + 5000) // 10000
    difference = gray.reshape(-1).astype(numpy.int64) - exact
    assert int(numpy.count_nonzero(difference)) == 19
    assert set(difference[difference != 0].tolist()) == {-1}


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
