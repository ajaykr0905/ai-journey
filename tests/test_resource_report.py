from copy import deepcopy

import pytest

from ai_journey.resource_accounting import (
    DecoderSpec,
    ResourceAccountingError,
    TrainingWorkload,
)
from ai_journey.resource_report import (
    ResourcePlan,
    build_resource_report,
    seal_resource_report,
    verify_resource_report,
)


def make_plan() -> ResourcePlan:
    return ResourcePlan(
        model=DecoderSpec(100, 2, 16, 4, 2, 32),
        workload=TrainingWorkload(5, 3, 2, 4, 10),
        parameter_dtype="bfloat16",
        gradient_dtype="bfloat16",
        optimizer_state_dtype="float32",
        activation_dtype="bfloat16",
        master_weights=True,
        checkpoint_segments=2,
        save_attention_probabilities=False,
        sharding_strategy="zero3",
        device_capacity_bytes=1_000_000,
        reserve_fraction=0.1,
        accelerator_peak_flops=1_000_000,
        model_flops_utilization=0.5,
        hourly_cost_per_accelerator=2.0,
    )


def test_resource_report_reconciles_compute_memory_runtime_and_cost():
    first = build_resource_report(make_plan())
    second = build_resource_report(make_plan())
    assert first == second
    assert first["schema_version"] == 1
    assert first["kind"] == "analytical_planning_estimate"
    assert first["measured"] is False
    assert first["parameters"]["count"] == 7_888
    assert first["memory"]["capacity"]["fits"] is True
    assert first["flops"]["training_total"] > 0
    assert first["runtime"]["seconds"] > 0
    assert first["cost"]["estimated_cost"] > 0
    assert len(first["limitations"]) == 3


def test_resource_report_seal_is_stable_and_does_not_mutate_input():
    report = build_resource_report(make_plan())
    first = seal_resource_report(report)
    second = seal_resource_report(report)
    assert first == second
    assert "integrity" not in report
    assert first["integrity"]["algorithm"] == "sha256"
    assert len(first["integrity"]["digest"]) == 64
    changed = build_resource_report(make_plan())
    changed["assumptions"]["reserve_fraction"] = 0.2
    assert (
        seal_resource_report(changed)["integrity"]["digest"]
        != first["integrity"]["digest"]
    )


def test_resource_report_verification_recomputes_every_derived_field():
    sealed = seal_resource_report(build_resource_report(make_plan()))
    assert verify_resource_report(sealed) is sealed

    tampered = deepcopy(sealed)
    tampered["memory"]["per_device"]["total"] += 1
    with pytest.raises(ResourceAccountingError, match="recomputation"):
        verify_resource_report(tampered)

    mislabeled = deepcopy(sealed)
    mislabeled["measured"] = True
    with pytest.raises(ResourceAccountingError, match="unmeasured"):
        verify_resource_report(mislabeled)
