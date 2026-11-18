from fractions import Fraction

import pytest

from ai_journey.matmul_model_traffic import (
    NO_REUSE_INTENSITY_LIMIT,
    TRAFFIC_MODELS,
    account_matmul,
    decimal_text,
    ideal_reuse_traffic,
    no_reuse_traffic,
    ratio_mapping,
)
from ai_journey.memory_access_tiling import MatmulShape

SHAPES = [
    MatmulShape(37, 45, 29),
    MatmulShape(32, 64, 16),
    MatmulShape(1, 1, 1),
    MatmulShape(1024, 1024, 1024),
    MatmulShape(4096, 1, 65536),
]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s.m}x{s.n}x{s.k}")
def test_both_models_count_flops_and_bytes_in_closed_form(shape):
    m, n, k = shape.m, shape.n, shape.k
    accounting = account_matmul(shape)
    no_reuse, ideal = accounting.no_reuse, accounting.ideal_reuse

    assert accounting.flops == no_reuse.flops == ideal.flops == 2 * m * n * k
    assert (no_reuse.model, ideal.model) == TRAFFIC_MODELS
    assert no_reuse.bytes == 4 * (2 * m * n * k + m * n)
    assert ideal.bytes == 4 * (m * k + k * n + m * n)
    assert no_reuse.intensity == Fraction(k, 4 * k + 2) < NO_REUSE_INTENSITY_LIMIT
    assert ideal.intensity == Fraction(m * n * k, 2 * (m * k + k * n + m * n))
    assert ideal.intensity >= no_reuse.intensity
    assert accounting.reuse_factor == Fraction(no_reuse.bytes, ideal.bytes)
    assert accounting.cross_checks() == {
        "memory_access_naive_traffic": True,
        "roofline_matmul_workload": True,
    }


def test_square_shapes_gain_intensity_only_from_reuse():
    for size in (32, 256, 1024, 4096):
        shape = MatmulShape(size, size, size)
        assert ideal_reuse_traffic(shape).intensity == Fraction(size, 6)
        assert no_reuse_traffic(shape).intensity == Fraction(size, 4 * size + 2)
    accounting = account_matmul(MatmulShape(1024, 1024, 1024)).to_mapping()

    assert accounting["no_reuse"] == {
        "a_elements": 2**30,
        "b_elements": 2**30,
        "c_elements": 2**20,
        "bytes": 4 * (2**31 + 2**20),
        "intensity": {"numerator": 512, "denominator": 2049, "decimal": "0.249878"},
    }
    assert accounting["ideal_reuse"]["intensity"] == {
        "numerator": 512,
        "denominator": 3,
        "decimal": "170.666667",
    }
    assert accounting["reuse_factor"]["decimal"] == "683.000000"
    assert accounting["no_reuse_intensity_limit"]["decimal"] == "0.250000"


def test_ratios_round_half_to_even_without_floats():
    assert decimal_text(Fraction(1, 8), 2) == "0.12"
    assert decimal_text(Fraction(3, 8), 2) == "0.38"
    assert decimal_text(Fraction(29, 118)) == "0.245763"
    assert decimal_text(Fraction(-1, 3)) == "-0.333333"
    assert decimal_text(Fraction(7)) == "7.000000"
    assert ratio_mapping(Fraction(6, 4)) == {
        "numerator": 3,
        "denominator": 2,
        "decimal": "1.500000",
    }
    with pytest.raises(TypeError, match="Fraction"):
        ratio_mapping(0.5)
    with pytest.raises(TypeError, match="MatmulShape"):
        account_matmul((2, 2, 2))
