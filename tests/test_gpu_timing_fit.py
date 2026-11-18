import itertools

import numpy
import pytest

from ai_journey.gpu_timing_clocks import GpuTimingError
from ai_journey.gpu_timing_fit import (
    FIT_METHOD,
    FIXED_DOMINANT,
    SIZE_DOMINANT,
    LineFit,
    OverheadSplit,
    fit_line,
    split,
)

FLOPS = (2, 1024, 65536, 2097152, 134217728)


def test_an_exact_line_is_recovered():
    medians = tuple(20_000 + 3 * x for x in FLOPS)

    fit = fit_line(FLOPS, medians)

    assert fit.intercept_ns == pytest.approx(20_000, rel=1e-12)
    assert fit.slope_ns_per_flop == pytest.approx(3, rel=1e-12)
    assert fit.points == 5
    assert fit.max_relative_residual < 1e-12
    assert fit.predict_ns(1000) == pytest.approx(23_000, rel=1e-12)


def test_the_fit_matches_numpy_weighted_least_squares():
    medians = (21_000, 19_500, 24_000, 400_000, 30_000_000)

    fit = fit_line(FLOPS, medians)

    x = numpy.array(FLOPS, dtype=numpy.float64)
    y = numpy.array(medians, dtype=numpy.float64)
    design = numpy.stack([numpy.ones_like(x), x], axis=1) / y[:, None]
    solution, *_ = numpy.linalg.lstsq(design, numpy.ones_like(y), rcond=None)
    assert fit.intercept_ns == pytest.approx(solution[0], rel=1e-6)
    assert fit.slope_ns_per_flop == pytest.approx(solution[1], rel=1e-6)
    residuals = numpy.abs(y - (solution[0] + solution[1] * x)) / y
    assert fit.max_relative_residual == pytest.approx(residuals.max(), rel=1e-6)


def test_the_fit_does_not_depend_on_point_order():
    medians = (21_000, 19_500, 24_000, 400_000, 30_000_000)
    first = fit_line(FLOPS, medians)

    for order in itertools.permutations(range(5)):
        other = fit_line(
            tuple(FLOPS[i] for i in order), tuple(medians[i] for i in order)
        )
        assert other.intercept_ns == first.intercept_ns
        assert other.slope_ns_per_flop == first.slope_ns_per_flop


def test_relative_weights_keep_the_intercept_near_the_small_points():
    # A 1 % error at the largest point would move an unweighted intercept by
    # far more than the small points' own 20 us.
    medians = (20_000, 20_100, 20_300, 30_000, 690_000)

    fit = fit_line(FLOPS, medians)

    assert 19_000 < fit.intercept_ns < 21_000


@pytest.mark.parametrize(
    ("flops", "medians", "message"),
    [
        ((1, 2), (1, 2), "at least 3"),
        ([1, 2, 3], (1, 2, 3), "at least 3"),
        ((1, 2, 3), (1, 2), "at least 3"),
        ((1, 2, 0), (1, 2, 3), "flops must hold positive"),
        ((1, 2, 3), (1, 2.0, 3), "medians_ns must hold positive"),
        ((1, 2, True), (1, 2, 3), "flops must hold positive"),
        ((4, 4, 4), (1, 2, 3), "two distinct"),
    ],
)
def test_bad_fit_inputs_are_refused(flops, medians, message):
    with pytest.raises(GpuTimingError, match=message):
        fit_line(flops, medians)


def test_a_split_puts_the_intercept_on_fixed_overhead():
    fit = LineFit(20_000.0, 0.001, 5, 0.1)

    small = split(25_000, fit)
    large = split(100_000, fit)

    assert small.to_mapping() == {
        "median_ns": 25_000,
        "fixed_ns": 20_000.0,
        "size_ns": 5_000.0,
        "fixed_share": 0.8,
        "regime": FIXED_DOMINANT,
    }
    assert large.size_ns == 80_000.0
    assert large.regime == SIZE_DOMINANT
    assert split(40_000, fit).regime == FIXED_DOMINANT


def test_a_negative_intercept_is_reported_as_fitted():
    result = split(1_000, LineFit(-50.0, 1.0, 3, 0.5))

    assert result.fixed_share == -0.05
    assert result.size_ns == 1_050.0
    assert result.regime == SIZE_DOMINANT


def test_bad_splits_are_refused():
    with pytest.raises(TypeError):
        split(1, (1.0, 1.0))
    with pytest.raises(GpuTimingError):
        OverheadSplit(0, 1.0)
    with pytest.raises(GpuTimingError):
        OverheadSplit(1, 1)
    with pytest.raises(GpuTimingError):
        OverheadSplit(1, float("nan"))


def test_the_method_text_names_the_weights_and_the_sums():
    assert "1 / median_ns**2" in FIT_METHOD
    assert "math.fsum" in FIT_METHOD
