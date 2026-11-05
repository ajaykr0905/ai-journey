import pytest

from ai_journey.scaling_experiment_analysis import ScalingAcceptancePolicy
from ai_journey.scaling_experiment_identity import ExperimentInputs, run_identity
from ai_journey.scaling_experiment_observation import ScalingObservation
from ai_journey.scaling_experiment_protocol import (
    ScalingExperimentProtocol,
    build_experiment_grid,
)
from ai_journey.scaling_experiment_report import build_experiment_report
from ai_journey.scaling_laws import ScalingLawError


def protocol():
    return ScalingExperimentProtocol(
        name="day-62-bounded-scaling",
        parameter_counts=(10, 20, 40),
        train_token_counts=(100, 200),
        seeds=(1, 2),
    )


def inputs():
    return ExperimentInputs(
        source_url="https://example.org/public.txt",
        source_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        code_revision="c" * 40,
    )


def observations():
    return tuple(
        ScalingObservation(
            run_id=run_identity(point, inputs()),
            point=point,
            validation_loss=10
            * (point.parameter_count * point.train_tokens * 6) ** -0.1,
            trained_tokens=point.train_tokens,
            optimizer_steps=10,
            elapsed_seconds=1.0,
        )
        for point in build_experiment_grid(protocol())
    )


def test_report_recomputes_fit_holdout_prediction_and_acceptance():
    report = build_experiment_report(
        protocol(),
        inputs(),
        observations(),
        evidence_kind="deterministic_reference_fixture",
        target_compute=48_000,
        policy=ScalingAcceptancePolicy(0.99, 0.01, 1.0, 4.0),
    )

    assert report["kind"] == "bounded_scaling_experiment"
    assert report["evidence_kind"] == "deterministic_reference_fixture"
    assert report["budget"]["run_count"] == 12
    assert len(report["observations"]) == 12
    assert len(report["aggregates"]) == 6
    assert report["fit"]["exponent"] < 0
    assert report["holdout"]["holdout_count"] == 2
    assert report["target"]["scope"]["classification"] == "bounded_extrapolation"
    assert report["acceptance"] == {"admitted": True, "reasons": ()}


def test_report_keeps_authentication_gpu_scale_and_learning_claims_false():
    report = build_experiment_report(
        protocol(),
        inputs(),
        observations(),
        evidence_kind="deterministic_reference_fixture",
        target_compute=48_000,
        policy=ScalingAcceptancePolicy(0.99, 0.01, 1.0, 4.0),
    )

    boundaries = report["evidence_boundaries"]
    assert boundaries["observations_supplied"] is True
    assert boundaries["observations_measured"] is False
    assert boundaries["execution_authenticated"] is False
    assert boundaries["gpu_execution_proven"] is False
    assert boundaries["large_scale_proven"] is False
    assert boundaries["learner_completion"] is False
    assert boundaries["residual_band_is_confidence_interval"] is False


def test_report_rejects_unlabelled_observation_provenance():
    with pytest.raises(ScalingLawError, match="evidence kind"):
        build_experiment_report(
            protocol(),
            inputs(),
            observations(),
            evidence_kind="unknown",
            target_compute=48_000,
            policy=ScalingAcceptancePolicy(0.99, 0.01, 1.0, 4.0),
        )
