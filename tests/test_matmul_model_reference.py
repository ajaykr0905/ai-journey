import math
from fractions import Fraction

import pytest

from ai_journey.kernel_launch_config import KernelLaunchError
from ai_journey.kernel_launch_reference import seeded_float32
from ai_journey.matmul_model_reference import (
    INPUT_STREAMS,
    MatmulModelError,
    correctly_rounded_matmul,
    matmul_inputs,
    order_rounding,
    reference_matmul,
    round_to_float32,
    ulp_distance,
)
from ai_journey.memory_access_tiling import MatmulShape


def test_operands_come_from_independent_seeded_float32_streams():
    shape = MatmulShape(m=3, n=5, k=4)
    a, b = matmul_inputs(74, shape)

    assert INPUT_STREAMS == ("matmul-a", "matmul-b")
    assert a == seeded_float32(74, "matmul-a", 12)
    assert b == seeded_float32(74, "matmul-b", 20)
    assert all(2.0**-8 <= abs(value) < 2.0**8 for value in a + b)
    # A longer stream starts with a shorter one, and each operand has its own.
    taller_a, same_b = matmul_inputs(74, MatmulShape(m=4, n=5, k=4))
    assert taller_a[:12] == a and same_b == b
    assert matmul_inputs(75, shape) != (a, b)


def test_the_reference_multiplies_row_major_operands():
    square = reference_matmul(
        [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], MatmulShape(2, 2, 2)
    )
    wide = reference_matmul(
        [1.0, -2.0], [0.5, 1.0, 2.0, 0.25, -1.0, 4.0], MatmulShape(1, 3, 2)
    )

    assert square == (19.0, 22.0, 43.0, 50.0)
    assert wide == (0.0, 3.0, -6.0)


def test_each_product_and_each_partial_sum_round_to_float32():
    # (1 + 2**-12)**2 = 1 + 2**-11 + 2**-24 rounds to 1 + 2**-11, a tie to
    # even, so the second term cancels it. One fused multiply-add keeps 2**-24.
    shape = MatmulShape(m=1, n=1, k=2)
    a = [1.0 + 2.0**-12, 1.0]
    b = [1.0 + 2.0**-12, -(1.0 + 2.0**-11)]
    assert reference_matmul(a, b, shape) == (0.0,)
    assert correctly_rounded_matmul(a, b, shape) == (2.0**-24,)

    # 2**24 + 1 is a tie that rounds to 2**24, so the stated order loses the 1.
    ordered = MatmulShape(m=1, n=1, k=3)
    terms = [1.0, 1.0, 1.0], [2.0**24, 1.0, -(2.0**24)]
    assert reference_matmul(*terms, ordered) == (0.0,)
    assert correctly_rounded_matmul(*terms, ordered) == (1.0,)
    reversed_terms = [1.0, 1.0, 1.0], [1.0, 2.0**24, -(2.0**24)]
    assert reference_matmul(*reversed_terms, ordered) == (0.0,)
    assert reference_matmul([1.0] * 3, [2.0**24, -(2.0**24), 1.0], ordered) == (1.0,)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Fraction(1) + Fraction(1, 2**24), 1.0),
        (Fraction(1) + Fraction(3, 2**24), 1.0 + 2.0**-22),
        (Fraction(1) + Fraction(2**24 + 1, 2**48), 1.0 + 2.0**-23),
        (Fraction(2**25 - 1, 2**25), 1.0),
        (Fraction(-1, 3), -float.fromhex("0x1.555556p-2")),
        (Fraction(10, 3), float.fromhex("0x1.aaaaaap+1")),
        (Fraction(2**-126), 2.0**-126),
        (Fraction(2**127) * (2 - Fraction(1, 2**23)), (2 - 2.0**-23) * 2.0**127),
        (Fraction(0), 0.0),
    ],
)
def test_round_to_float32_rounds_ties_to_even_in_integers(value, expected):
    result = round_to_float32(value)

    assert result == expected
    assert math.copysign(1.0, result) == math.copysign(1.0, expected)


def test_round_to_float32_refuses_values_outside_the_normal_range():
    for value in (Fraction(2**128), Fraction(1, 2**127), -Fraction(2**129)):
        with pytest.raises(MatmulModelError, match="normal float32 range"):
            round_to_float32(value)
    with pytest.raises(TypeError, match="Fraction"):
        round_to_float32(0.5)


def test_one_term_dot_products_round_once_in_both_functions():
    shape = MatmulShape(m=6, n=7, k=1)
    a, b = matmul_inputs(3, shape)

    assert reference_matmul(a, b, shape) == correctly_rounded_matmul(a, b, shape)


def test_the_stated_order_moves_most_seeded_elements_off_the_exact_result():
    shape = MatmulShape(m=7, n=9, k=29)
    a, b = matmul_inputs(74, shape)
    stated = reference_matmul(a, b, shape)
    exact = correctly_rounded_matmul(a, b, shape)
    rounding = order_rounding(stated, exact)

    # Pinned so a change to the streams or the rule cannot pass unnoticed.
    # Terms of both signs cancel, and a small sum has a small ulp.
    assert rounding.to_mapping() == {
        "elements": 63,
        "differing": 46,
        "max_ulp_difference": 17,
    }
    assert rounding.differing == sum(x != y for x, y in zip(stated, exact))
    assert rounding.max_ulp_difference == max(
        ulp_distance(x, y) for x, y in zip(stated, exact)
    )


def test_ulp_distance_counts_float32_steps_across_zero():
    assert ulp_distance(1.0, 1.0 + 2.0**-23) == 1
    assert ulp_distance(1.0 - 2.0**-24, 1.0) == 1
    assert ulp_distance(-0.0, 0.0) == 0
    assert ulp_distance(-(2.0**-149), 2.0**-149) == 2
    assert ulp_distance(-1.0, 1.0) == 2 * 0x3F800000
    with pytest.raises(KernelLaunchError, match="float32"):
        ulp_distance(0.1, 0.2)


def test_operands_must_match_the_shape_and_hold_float32_values():
    shape = MatmulShape(m=2, n=2, k=2)

    with pytest.raises(MatmulModelError, match="m \\* k values"):
        reference_matmul([1.0] * 3, [1.0] * 4, shape)
    with pytest.raises(MatmulModelError, match="k \\* n values"):
        correctly_rounded_matmul([1.0] * 4, [1.0] * 5, shape)
    with pytest.raises(KernelLaunchError, match="round it with to_float32"):
        reference_matmul([0.1] * 4, [1.0] * 4, shape)
    with pytest.raises(KernelLaunchError, match="round it with to_float32"):
        reference_matmul([math.inf] * 4, [1.0] * 4, shape)
    with pytest.raises(TypeError, match="MatmulShape"):
        matmul_inputs(74, (2, 2, 2))
    with pytest.raises(MatmulModelError, match="equal, nonzero length"):
        order_rounding([1.0], [1.0, 2.0])
    with pytest.raises(MatmulModelError, match="equal, nonzero length"):
        order_rounding([], [])
