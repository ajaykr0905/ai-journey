import math
from fractions import Fraction

import pytest

from ai_journey.matmul_model_reference import MatmulModelError
from ai_journey.matmul_model_roofline import (
    BOUNDS,
    Ceilings,
    parse_decimal,
    roofline,
)
from ai_journey.matmul_model_traffic import account_matmul
from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.roofline import KernelWorkload, RooflineHardware, analyze_roofline

RIDGE_4 = Ceilings("declared-ridge-4", "4", "1")


def test_below_the_ridge_the_bandwidth_ceiling_sets_the_rate():
    point = roofline(Fraction(1, 4), RIDGE_4)

    assert RIDGE_4.ridge_point == 4
    assert (point.attainable_gflops, point.bound) == (Fraction(1, 4), "memory")
    assert point.fraction_of_compute_ceiling == Fraction(1, 16)
    assert point.to_mapping() == {
        "intensity": {"numerator": 1, "denominator": 4, "decimal": "0.250000"},
        "ridge_point": {"numerator": 4, "denominator": 1, "decimal": "4.000000"},
        "attainable_gflops": {"numerator": 1, "denominator": 4, "decimal": "0.250000"},
        "bound": "memory",
        "fraction_of_compute_ceiling": {
            "numerator": 1,
            "denominator": 16,
            "decimal": "0.062500",
        },
    }


def test_at_and_above_the_ridge_the_compute_ceiling_sets_the_rate():
    ceilings = Ceilings("example", "19.5", "1.5")
    ridge = roofline(Fraction(13), ceilings)
    above = roofline(Fraction(512, 3), ceilings)

    assert ceilings.ridge_point == 13
    assert (ridge.bound, ridge.attainable_gflops) == ("ridge", Fraction(39, 2))
    assert (above.bound, above.attainable_gflops) == ("compute", Fraction(39, 2))
    assert above.fraction_of_compute_ceiling == 1
    assert BOUNDS == ("memory", "ridge", "compute")


def test_the_naive_kernel_without_reuse_reaches_under_a_quarter_of_the_ridge():
    # Below the ridge the fraction of the compute ceiling is intensity/ridge.
    for ceilings in (RIDGE_4, Ceilings("declared-ridge-16", "16", "1")):
        for size in (32, 1024):
            accounting = account_matmul(MatmulShape(size, size, size))
            point = roofline(accounting.no_reuse.intensity, ceilings)
            assert point.bound == "memory"
            assert point.fraction_of_compute_ceiling == (
                accounting.no_reuse.intensity / ceilings.ridge_point
            )
            assert (
                point.fraction_of_compute_ceiling
                < Fraction(1, 4) / ceilings.ridge_point
            )


@pytest.mark.parametrize(
    "intensity", [Fraction(1, 8), Fraction(29, 118), Fraction(3), Fraction(1000)]
)
@pytest.mark.parametrize(
    ("compute", "bandwidth"), [("4", "1"), ("19.5", "1.5"), ("1000", "0.25")]
)
def test_exact_points_agree_with_the_float_roofline_module(
    intensity, compute, bandwidth
):
    ceilings = Ceilings("check", compute, bandwidth)
    point = roofline(intensity, ceilings)
    # 2 x numerator FLOPs over 2 x denominator bytes is the same intensity.
    workload = KernelWorkload(
        "check", 2 * intensity.numerator, intensity.denominator, intensity.denominator
    )
    hardware = RooflineHardware(float(compute) * 1e9, float(bandwidth) * 1e9)
    estimate = analyze_roofline(workload, hardware)

    assert estimate.bottleneck == point.bound
    assert math.isclose(
        estimate.attainable_flops_per_second / 1e9,
        float(point.attainable_gflops),
        rel_tol=1e-12,
    )
    assert math.isclose(
        hardware.ridge_point, float(ceilings.ridge_point), rel_tol=1e-12
    )


@pytest.mark.parametrize(
    "text",
    ["019.5", "1e3", "1.50", "", "-1", "0", "0.000", " 1", "1.", ".5", "1.1234567"],
)
def test_ceilings_are_positive_canonical_decimal_strings(text):
    with pytest.raises(MatmulModelError, match="decimal string|positive"):
        parse_decimal(text, "compute_gflops")


def test_ceilings_parse_exact_fields_and_refuse_floats():
    assert parse_decimal("0.25", "x") == Fraction(1, 4)
    assert parse_decimal("123456789012.12345", "x") == Fraction("123456789012.12345")
    mapping = {
        "label": "measured-host",
        "compute_gflops": "19.5",
        "bandwidth_gbps": "100",
    }
    assert Ceilings.from_mapping(mapping).to_mapping() == mapping
    for broken in (
        {**mapping, "extra": 1},
        {key: mapping[key] for key in ("label", "compute_gflops")},
    ):
        with pytest.raises(MatmulModelError, match="exactly label"):
            Ceilings.from_mapping(broken)
    with pytest.raises(MatmulModelError, match="decimal string"):
        Ceilings("x", 19.5, "1")
    with pytest.raises(MatmulModelError, match="lowercase slug"):
        Ceilings("Measured", "1", "1")
    with pytest.raises(MatmulModelError, match="positive Fraction"):
        roofline(0.25, RIDGE_4)
    with pytest.raises(MatmulModelError, match="positive Fraction"):
        roofline(Fraction(0), RIDGE_4)
    with pytest.raises(TypeError, match="Ceilings"):
        roofline(Fraction(1), mapping)
