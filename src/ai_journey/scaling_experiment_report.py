"""Build claim-bounded scaling experiment reports from complete observations."""

from __future__ import annotations

from dataclasses import asdict

from .scaling_experiment_analysis import (
    ScalingAcceptancePolicy,
    aggregate_replicates,
    assess_scaling_acceptance,
    classify_extrapolation,
    evaluate_fit_residuals,
    evaluate_holdout,
    fit_compute_loss,
    largest_model_holdout,
    residual_prediction_band,
)
from .scaling_experiment_budget import estimate_experiment_budget
from .scaling_experiment_identity import ExperimentInputs
from .scaling_experiment_observation import (
    ScalingObservation,
    validate_observation_set,
)
from .scaling_experiment_protocol import ScalingExperimentProtocol


def build_experiment_report(
    protocol: ScalingExperimentProtocol,
    inputs: ExperimentInputs,
    observations: tuple[ScalingObservation, ...],
    *,
    target_compute: float,
    policy: ScalingAcceptancePolicy,
) -> dict[str, object]:
    """Recompute every analytical field from identity-verified observations."""

    ordered = validate_observation_set(protocol, inputs, observations)
    aggregates = aggregate_replicates(ordered)
    split = largest_model_holdout(aggregates)
    fit = fit_compute_loss(split.training)
    residuals = evaluate_fit_residuals(split.training, fit)
    holdout = evaluate_holdout(fit, residuals, split.holdout)
    scope = classify_extrapolation(fit, target_compute)
    band = residual_prediction_band(fit, residuals, target_compute)
    acceptance = assess_scaling_acceptance(fit, holdout, scope, policy)
    budget = estimate_experiment_budget(protocol)
    return {
        "schema_version": 1,
        "kind": "bounded_scaling_experiment",
        "protocol": asdict(protocol),
        "inputs": asdict(inputs),
        "observations": [
            {
                "run_id": item.run_id,
                "point": asdict(item.point),
                "validation_loss": item.validation_loss,
                "trained_tokens": item.trained_tokens,
                "optimizer_steps": item.optimizer_steps,
                "elapsed_seconds": item.elapsed_seconds,
            }
            for item in ordered
        ],
        "budget": asdict(budget),
        "aggregates": [asdict(item) for item in aggregates],
        "fit": asdict(fit),
        "training_residuals": asdict(residuals),
        "holdout": asdict(holdout),
        "target": {
            "compute": float(target_compute),
            "scope": asdict(scope),
            "prediction_band": asdict(band),
        },
        "policy": asdict(policy),
        "acceptance": asdict(acceptance),
        "evidence_boundaries": {
            "observations_supplied": True,
            "execution_authenticated": False,
            "gpu_execution_proven": False,
            "large_scale_proven": False,
            "learner_completion": False,
            "residual_band_is_confidence_interval": False,
        },
    }
