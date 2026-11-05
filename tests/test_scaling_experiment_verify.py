from copy import deepcopy

import pytest

from ai_journey.scaling_experiment_analysis import ScalingAcceptancePolicy
from ai_journey.scaling_experiment_report import build_experiment_report
from ai_journey.scaling_experiment_verify import (
    seal_experiment_report,
    verify_experiment_report,
)
from ai_journey.scaling_laws import ScalingLawError
from test_scaling_experiment_report import inputs, observations, protocol


def report():
    return build_experiment_report(
        protocol(),
        inputs(),
        observations(),
        target_compute=48_000,
        policy=ScalingAcceptancePolicy(0.99, 0.01, 1.0, 4.0),
    )


def test_experiment_report_seal_is_stable_and_nonmutating():
    unsigned = report()

    first = seal_experiment_report(unsigned)
    second = seal_experiment_report(unsigned)

    assert first == second
    assert "integrity" not in unsigned
    assert len(first["integrity"]["digest"]) == 64
    assert verify_experiment_report(first) is first


def test_verifier_rejects_tampering_even_after_resealing():
    changed = deepcopy(report())
    changed["holdout"]["mean_absolute_percentage_error"] = 0.5

    with pytest.raises(ScalingLawError, match="integrity"):
        verify_experiment_report(
            {**changed, "integrity": {"algorithm": "sha256", "digest": "0" * 64}}
        )

    with pytest.raises(ScalingLawError, match="recomputation"):
        verify_experiment_report(seal_experiment_report(changed))
