"""Fit a simple observed power law without hiding residual error.

This module fits ``y = coefficient * x ** exponent`` in log space. It does not
fit an irreducible-loss offset and must not be used to claim extrapolation
quality outside the supplied positive observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .scaling_laws import ScalingLawError


@dataclass(frozen=True)
class PowerLawFit:
    """A log-space least-squares fit of y = coefficient * x ** exponent.

    It keeps the observed x range and log-space R-squared so later gates can
    tell interpolation from extrapolation and reject weak fits.
    """

    coefficient: float
    exponent: float
    r_squared_log_space: float
    observation_count: int
    minimum_x: float
    maximum_x: float

    def predict(self, x: float) -> float:
        """Return the fitted y at a positive finite x, without any range check.

        classify_extrapolation judges distance beyond maximum_x; non-numeric x
        raises TypeError and nonpositive x raises ScalingLawError.
        """

        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError("x must be numeric")
        if not math.isfinite(x) or x <= 0:
            raise ScalingLawError("x must be positive and finite")
        return self.coefficient * float(x) ** self.exponent


def _positive_observations(values, name: str) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) < 3:
        raise ScalingLawError(f"{name} must contain at least three observations")
    result = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ScalingLawError(f"{name} must be positive and finite")
        result.append(float(value))
    return tuple(result)


def fit_power_law(
    x_values: tuple[float, ...], y_values: tuple[float, ...]
) -> PowerLawFit:
    """Fit a log-space ordinary least-squares power law."""

    xs = _positive_observations(x_values, "x_values")
    ys = _positive_observations(y_values, "y_values")
    if len(xs) != len(ys):
        raise ScalingLawError("x_values and y_values must have equal length")
    if len(set(xs)) < 2:
        raise ScalingLawError("x_values must contain at least two unique values")
    log_x = tuple(math.log(value) for value in xs)
    log_y = tuple(math.log(value) for value in ys)
    mean_x = math.fsum(log_x) / len(log_x)
    mean_y = math.fsum(log_y) / len(log_y)
    centered_x = tuple(value - mean_x for value in log_x)
    centered_y = tuple(value - mean_y for value in log_y)
    denominator = math.fsum(value * value for value in centered_x)
    exponent = (
        math.fsum(
            left * right for left, right in zip(centered_x, centered_y, strict=True)
        )
        / denominator
    )
    intercept = mean_y - exponent * mean_x
    predicted = tuple(intercept + exponent * value for value in log_x)
    residual = math.fsum(
        (actual - estimate) ** 2
        for actual, estimate in zip(log_y, predicted, strict=True)
    )
    total = math.fsum(value * value for value in centered_y)
    r_squared = 1.0 if total == 0 and residual == 0 else 1 - residual / total
    return PowerLawFit(
        coefficient=math.exp(intercept),
        exponent=exponent,
        r_squared_log_space=r_squared,
        observation_count=len(xs),
        minimum_x=min(xs),
        maximum_x=max(xs),
    )
