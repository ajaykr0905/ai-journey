"""Transparent analysis for measured scaling experiment observations."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics

from .scaling_experiment_budget import point_training_flops
from .scaling_experiment_observation import ScalingObservation
from .scaling_experiment_protocol import ExperimentPoint
from .scaling_fit import PowerLawFit, fit_power_law
from .scaling_laws import ScalingLawError


@dataclass(frozen=True, order=True)
class AggregatedPoint:
    parameter_count: int
    train_tokens: int
    mean_validation_loss: float
    sample_standard_deviation: float
    replicate_count: int


@dataclass(frozen=True)
class FitSplit:
    training: tuple[AggregatedPoint, ...]
    holdout: tuple[AggregatedPoint, ...]


@dataclass(frozen=True)
class FitResiduals:
    mean_absolute_error: float
    mean_absolute_percentage_error: float
    root_mean_squared_log_error: float
    observation_count: int


@dataclass(frozen=True)
class ExtrapolationScope:
    target_compute: float
    factor_beyond_observed_maximum: float
    classification: str


@dataclass(frozen=True)
class ResidualPredictionBand:
    estimate: float
    lower: float
    upper: float
    log_error_multiplier: float


@dataclass(frozen=True)
class HoldoutEvaluation:
    mean_absolute_percentage_error: float
    within_residual_band_fraction: float
    holdout_count: int


def aggregate_replicates(
    observations: tuple[ScalingObservation, ...],
) -> tuple[AggregatedPoint, ...]:
    """Aggregate seeds by coordinate without pooling different model/token pairs."""

    if not isinstance(observations, tuple) or not observations:
        raise ScalingLawError("observations must be a nonempty tuple")
    if any(not isinstance(item, ScalingObservation) for item in observations):
        raise TypeError("observations must contain ScalingObservation values")
    groups: dict[tuple[int, int], list[float]] = {}
    for item in observations:
        key = (item.point.parameter_count, item.point.train_tokens)
        groups.setdefault(key, []).append(float(item.validation_loss))
    if any(len(losses) < 2 for losses in groups.values()):
        raise ScalingLawError("every coordinate requires at least two replicates")
    return tuple(
        AggregatedPoint(
            parameter_count=key[0],
            train_tokens=key[1],
            mean_validation_loss=statistics.fmean(losses),
            sample_standard_deviation=statistics.stdev(losses),
            replicate_count=len(losses),
        )
        for key, losses in sorted(groups.items())
    )


def largest_model_holdout(points: tuple[AggregatedPoint, ...]) -> FitSplit:
    """Hold out the largest model independently at each token budget."""

    if not isinstance(points, tuple) or not points:
        raise ScalingLawError("points must be a nonempty tuple")
    if any(not isinstance(item, AggregatedPoint) for item in points):
        raise TypeError("points must contain AggregatedPoint values")
    if len({(item.parameter_count, item.train_tokens) for item in points}) != len(
        points
    ):
        raise ScalingLawError("points must have unique coordinates")
    largest_by_tokens = {
        tokens: max(
            item.parameter_count for item in points if item.train_tokens == tokens
        )
        for tokens in {item.train_tokens for item in points}
    }
    holdout = tuple(
        sorted(
            item
            for item in points
            if item.parameter_count == largest_by_tokens[item.train_tokens]
        )
    )
    training = tuple(sorted(set(points) - set(holdout)))
    if len(training) < 3:
        raise ScalingLawError("holdout split requires at least three training points")
    return FitSplit(training=training, holdout=holdout)


def fit_compute_loss(
    training: tuple[AggregatedPoint, ...],
    *,
    flops_per_parameter_token: int = 6,
) -> PowerLawFit:
    """Fit mean validation loss against declared analytical training compute."""

    if not isinstance(training, tuple) or len(training) < 3:
        raise ScalingLawError("training must contain at least three points")
    compute = tuple(
        float(
            point_training_flops(
                ExperimentPoint(row.parameter_count, row.train_tokens, 0),
                flops_per_parameter_token=flops_per_parameter_token,
            )
        )
        for row in training
    )
    losses = tuple(row.mean_validation_loss for row in training)
    return fit_power_law(compute, losses)


def evaluate_fit_residuals(
    points: tuple[AggregatedPoint, ...], fit: PowerLawFit
) -> FitResiduals:
    """Report absolute, relative, and log-space error on supplied points."""

    if not isinstance(points, tuple) or not points:
        raise ScalingLawError("points must be a nonempty tuple")
    if not isinstance(fit, PowerLawFit):
        raise TypeError("fit must be PowerLawFit")
    actual = tuple(row.mean_validation_loss for row in points)
    predicted = tuple(
        fit.predict(float(row.parameter_count * row.train_tokens * 6)) for row in points
    )
    absolute = tuple(
        abs(left - right) for left, right in zip(actual, predicted, strict=True)
    )
    log_squared = tuple(
        (math.log(left) - math.log(right)) ** 2
        for left, right in zip(actual, predicted, strict=True)
    )
    return FitResiduals(
        mean_absolute_error=statistics.fmean(absolute),
        mean_absolute_percentage_error=statistics.fmean(
            error / value for error, value in zip(absolute, actual, strict=True)
        ),
        root_mean_squared_log_error=math.sqrt(statistics.fmean(log_squared)),
        observation_count=len(points),
    )


def classify_extrapolation(
    fit: PowerLawFit, target_compute: float, *, bounded_factor: float = 4.0
) -> ExtrapolationScope:
    """Classify predictions by distance beyond the observed compute range."""

    if not isinstance(fit, PowerLawFit):
        raise TypeError("fit must be PowerLawFit")
    for value, name in (
        (target_compute, "target_compute"),
        (bounded_factor, "bounded_factor"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ScalingLawError(f"{name} must be positive and finite")
    factor = float(target_compute) / fit.maximum_x
    classification = (
        "interpolation"
        if factor <= 1
        else "bounded_extrapolation"
        if factor <= bounded_factor
        else "speculative_extrapolation"
    )
    return ExtrapolationScope(float(target_compute), factor, classification)


def residual_prediction_band(
    fit: PowerLawFit,
    residuals: FitResiduals,
    target_compute: float,
    *,
    log_error_multiplier: float = 1.96,
) -> ResidualPredictionBand:
    """Build a descriptive log-residual band, not a confidence interval."""

    scope = classify_extrapolation(fit, target_compute)
    if not isinstance(residuals, FitResiduals):
        raise TypeError("residuals must be FitResiduals")
    if (
        isinstance(log_error_multiplier, bool)
        or not isinstance(log_error_multiplier, (int, float))
        or not math.isfinite(log_error_multiplier)
        or log_error_multiplier <= 0
    ):
        raise ScalingLawError("log_error_multiplier must be positive and finite")
    estimate = fit.predict(scope.target_compute)
    spread = math.exp(
        float(log_error_multiplier) * residuals.root_mean_squared_log_error
    )
    return ResidualPredictionBand(
        estimate=estimate,
        lower=estimate / spread,
        upper=estimate * spread,
        log_error_multiplier=float(log_error_multiplier),
    )


def evaluate_holdout(
    fit: PowerLawFit,
    residuals: FitResiduals,
    holdout: tuple[AggregatedPoint, ...],
) -> HoldoutEvaluation:
    """Evaluate predeclared holdout error without refitting on held-out rows."""

    if not isinstance(holdout, tuple) or not holdout:
        raise ScalingLawError("holdout must be a nonempty tuple")
    errors = []
    within = 0
    for row in holdout:
        compute = float(row.parameter_count * row.train_tokens * 6)
        band = residual_prediction_band(fit, residuals, compute)
        errors.append(
            abs(row.mean_validation_loss - band.estimate) / row.mean_validation_loss
        )
        within += band.lower <= row.mean_validation_loss <= band.upper
    return HoldoutEvaluation(
        mean_absolute_percentage_error=statistics.fmean(errors),
        within_residual_band_fraction=within / len(holdout),
        holdout_count=len(holdout),
    )
