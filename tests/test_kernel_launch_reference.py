import math
import struct
from fractions import Fraction
from hashlib import sha256

import pytest

from ai_journey.kernel_launch_config import KernelLaunchError
from ai_journey.kernel_launch_reference import (
    GRAYSCALE_RULE,
    GRAYSCALE_WEIGHTS,
    GRAYSCALE_WEIGHTS_SOURCE,
    compare_exact,
    grayscale_rounding,
    reference_rgb_to_grayscale,
    reference_vector_add,
    rgb_image,
    rounded_sums,
    seeded_bytes,
    seeded_float32,
    seeded_uint8,
    vector_add_inputs,
)


def tie_triples():
    """Return every RGB triple whose exact weighted sum ends in exactly .5.

    1140 b = 5000 - 2989 r - 5870 g (mod 10000) is solvable only when the right
    side is a multiple of gcd(1140, 10000) = 20; then 57 b = rest / 20 (mod 500).
    """

    inverse = pow(57, -1, 500)
    ties = []
    for red in range(256):
        for green in range(256):
            rest = (5000 - 2989 * red - 5870 * green) % 10000
            if rest % 20 == 0:
                blue = (rest // 20) * inverse % 500
                if blue < 256:
                    ties.append((red, green, blue))
    return ties


def test_seeded_streams_are_pinned_independent_and_prefix_stable():
    first = seeded_bytes(73, "rgb-image", 3600)

    assert first[:8].hex() == "508130ece898541e"
    assert sha256(first).hexdigest() == (
        "d2b4f71da860d52c25f78508ee69ebfa2c7a500ae40ccb378c7a5d7c37832e9b"
    )
    assert seeded_bytes(73, "rgb-image", 33) == first[:33]
    assert seeded_bytes(74, "rgb-image", 32) != first[:32]
    assert seeded_bytes(73, "other", 32) != first[:32]
    assert seeded_uint8(73, "rgb-image", 4) == tuple(first[:4])
    assert rgb_image(73, 40, 30) == tuple(first)


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: seeded_bytes(-1, "a", 1), KernelLaunchError, "input seed"),
        (lambda: seeded_bytes(2**32, "a", 1), KernelLaunchError, "input seed"),
        (lambda: seeded_bytes(True, "a", 1), TypeError, "input seed"),
        (lambda: seeded_bytes(1, "A", 1), KernelLaunchError, "lowercase slug"),
        (lambda: seeded_bytes(1, "a", 0), KernelLaunchError, "count"),
        (lambda: rgb_image(1, 4096, 4096), KernelLaunchError, "at most"),
    ],
)
def test_seeded_streams_reject_ambiguous_requests(call, error, message):
    with pytest.raises(error, match=message):
        call()


def test_seeded_float32_values_are_exact_and_spread_over_sixteen_binades():
    a, b = vector_add_inputs(73, 1000)

    assert a[:2] == (0.1866776943206787, 0.005656493827700615)
    assert a != b
    for value in a + b:
        assert struct.unpack("<f", struct.pack("<f", value))[0] == value
        assert 2**-8 <= abs(value) < 2**8
    exponents = {math.frexp(abs(value))[1] for value in a + b}
    assert exponents == set(range(-7, 9))
    assert 0 < sum(value < 0 for value in a) < 1000
    assert seeded_float32(73, "vector-add-a", 3) == a[:3]


def test_float32_reference_rounds_each_sum_once_to_nearest_even():
    # 1 + 2**-24 lies halfway between 1 and the next float32, so it rounds to
    # the even 1.0; 1 + 3 * 2**-24 lies halfway too and rounds up to even.
    assert reference_vector_add([1.0, 1.0 + 2**-23, 0.5], [2**-24, 2**-24, 0.25]) == (
        1.0,
        1.0 + 2**-22,
        0.75,
    )
    a, b = vector_add_inputs(73, 1000)
    expected = tuple(
        struct.unpack("<f", struct.pack("<f", x + y))[0] for x, y in zip(a, b)
    )
    assert reference_vector_add(a, b) == expected
    assert rounded_sums(a, b) == 851
    assert rounded_sums([0.5], [0.25]) == 0
    with pytest.raises(KernelLaunchError, match="float32 values"):
        reference_vector_add([0.1], [0.0])
    with pytest.raises(KernelLaunchError, match="finite floats"):
        reference_vector_add([1], [0.0])
    with pytest.raises(KernelLaunchError, match="equal, nonzero length"):
        reference_vector_add([0.5], [0.5, 0.5])


def test_grayscale_weights_rule_and_source_are_stated_exactly():
    weights = [Fraction(weight) for weight in GRAYSCALE_WEIGHTS]

    assert GRAYSCALE_WEIGHTS == ("0.2989", "0.5870", "0.1140")
    assert [weight * 10_000 for weight in weights] == [2989, 5870, 1140]
    assert sum(weights) == Fraction(9999, 10_000)
    for weight, luma in zip(weights, ("0.299", "0.587", "0.114")):
        assert abs(weight - Fraction(luma)) <= Fraction(1, 10_000)
    assert "GPU MODE Lecture 3" in GRAYSCALE_WEIGHTS_SOURCE
    assert "BT.601" in GRAYSCALE_WEIGHTS_SOURCE
    assert "half up" in GRAYSCALE_RULE and "[0, 255]" in GRAYSCALE_RULE


def test_reference_grayscale_rounds_ties_up_and_never_needs_the_clamp():
    image = [0, 0, 250, 0, 36, 12, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0]

    # 0.1140 * 250 = 28.5 and 0.5870 * 36 + 0.1140 * 12 = 22.5 are ties.
    assert reference_rgb_to_grayscale(image, 3, 2) == (29, 23, 255, 0, 76, 150)
    assert grayscale_rounding(image, 3, 2) == {"ties": 2, "truncation_differs": 4}
    with pytest.raises(KernelLaunchError, match="3 \\* width \\* height"):
        reference_rgb_to_grayscale(image, 2, 2)
    with pytest.raises(KernelLaunchError, match="uint8 integers"):
        reference_rgb_to_grayscale([256, 0, 0], 1, 1)


def test_the_exact_rule_differs_from_a_float_formula_on_292_ties():
    ties = tie_triples()

    assert len(ties) == 1703
    for red, green, blue in ties:
        assert (2989 * red + 5870 * green + 1140 * blue) % 10_000 == 5000
    flat = [channel for triple in ties for channel in triple]
    rule = reference_rgb_to_grayscale(flat, len(ties), 1)
    double = [
        math.floor(0.2989 * red + 0.5870 * green + 0.1140 * blue + 0.5)
        for red, green, blue in ties
    ]
    assert sum(left != right for left, right in zip(rule, double)) == 292
    assert all(left - right in (0, 1) for left, right in zip(rule, double))


def test_seeded_image_rounding_counts_are_pinned():
    image = rgb_image(73, 40, 30)

    assert grayscale_rounding(image, 40, 30) == {"ties": 0, "truncation_differs": 612}
    assert len(reference_rgb_to_grayscale(image, 40, 30)) == 1200


def test_exact_comparison_reports_bit_level_mismatches():
    same = compare_exact("float32", [0.5, -2.0], [0.5, -2.0])
    signed = compare_exact("float32", [0.0, 1.0, 0.0], [0.0, 1.0, -0.0])
    pixels = compare_exact("uint8", [1, 2, 3, 4], [1, 9, 3, 8])

    assert same.equal and same.first_mismatch is None
    assert (signed.mismatches, signed.first_mismatch) == (1, 2)
    assert pixels.to_mapping() == {
        "dtype": "uint8",
        "elements": 4,
        "mismatches": 2,
        "first_mismatch": 1,
        "equal": False,
    }
    with pytest.raises(KernelLaunchError, match="differ in length"):
        compare_exact("uint8", [1], [1, 2])
    with pytest.raises(KernelLaunchError, match="round it with to_float32"):
        compare_exact("float32", [0.1], [0.1])
    with pytest.raises(KernelLaunchError, match="dtype must be one of"):
        compare_exact("int8", [1], [1])
