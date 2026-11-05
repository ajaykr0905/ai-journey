import pytest

from ai_journey.scaling_fit import fit_power_law
from ai_journey.scaling_laws import ScalingLawError


def test_power_law_fit_recovers_an_exact_observed_curve():
    fit = fit_power_law((1, 2, 4, 8), (8, 4, 2, 1))

    assert fit.coefficient == pytest.approx(8)
    assert fit.exponent == pytest.approx(-1)
    assert fit.r_squared_log_space == pytest.approx(1)
    assert fit.predict(16) == pytest.approx(0.5)
    assert fit.observation_count == 4
    assert fit.minimum_x == 1
    assert fit.maximum_x == 8


def test_power_law_fit_supports_explicit_interpolation_without_extrapolation():
    fit = fit_power_law((1, 2, 4, 8), (8, 4, 2, 1))

    assert fit.predict_within_observed_range(1) == pytest.approx(8)
    assert fit.predict_within_observed_range(3) == pytest.approx(8 / 3)
    assert fit.predict_within_observed_range(8) == pytest.approx(1)
    with pytest.raises(ScalingLawError, match="outside the observed fit range"):
        fit.predict_within_observed_range(16)


def test_power_law_fit_reports_imperfect_log_space_residuals():
    fit = fit_power_law((1, 2, 4), (8, 5, 2))

    assert 0 < fit.r_squared_log_space < 1
    assert fit.exponent < 0


@pytest.mark.parametrize(
    "xs,ys,message",
    [
        ((1, 2), (2, 1), "at least three"),
        ((1, 1, 1), (3, 2, 1), "two unique"),
        ((1, 2, 3), (3, 0, 1), "positive and finite"),
        ((1, 2, 3), (3, 2, 1, 0.5), "equal length"),
    ],
)
def test_power_law_fit_rejects_insufficient_or_invalid_observations(xs, ys, message):
    with pytest.raises(ScalingLawError, match=message):
        fit_power_law(xs, ys)


def test_power_law_prediction_rejects_extrapolation_coordinates_that_are_invalid():
    fit = fit_power_law((1, 2, 4), (8, 4, 2))

    with pytest.raises(ScalingLawError, match="positive and finite"):
        fit.predict(0)


@pytest.mark.parametrize(
    "ys",
    [
        (1.0, 1e100, 1e200),
        (1.0, 1e-100, 1e-200),
    ],
)
def test_power_law_prediction_rejects_overflow_and_underflow(ys):
    fit = fit_power_law((1, 2, 4), ys)

    with pytest.raises(ScalingLawError, match="prediction must be positive and finite"):
        fit.predict(1e308)
