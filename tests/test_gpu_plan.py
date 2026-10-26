"""Memory admission must distinguish analytical fit from observed availability."""

from copy import deepcopy

import pytest

from ai_journey.gpu_plan import admit_gpu_resources
from ai_journey.gpu_preflight import GpuTopology
from ai_journey.resource_accounting import DecoderSpec, TrainingWorkload
from ai_journey.resource_report import (
    ResourcePlan,
    build_resource_report,
    seal_resource_report,
)


def resource_report():
    plan = ResourcePlan(
        model=DecoderSpec(256, 2, 16, 4, 4, 32, True),
        workload=TrainingWorkload(8, 2, 4, 2, 10),
        parameter_dtype="bfloat16",
        gradient_dtype="bfloat16",
        optimizer_state_dtype="float32",
        activation_dtype="bfloat16",
        master_weights=True,
        checkpoint_segments=1,
        save_attention_probabilities=True,
        sharding_strategy="ddp",
        device_capacity_bytes=1_000_000,
        reserve_fraction=0.1,
        accelerator_peak_flops=1_000_000,
        model_flops_utilization=0.5,
        hourly_cost_per_accelerator=0,
    )
    return seal_resource_report(build_resource_report(plan))


def simulated_probe():
    return {
        "schema_version": 1,
        "kind": "nvidia_host_probe",
        "observed_at_utc": "2026-10-26T00:00:00+00:00",
        "system": "Linux",
        "machine": "x86_64",
        "torch": "test",
        "cuda_runtime": "12.4",
        "nccl_available": True,
        "visible_devices_sha256": None,
        "available": True,
        "reason": None,
        "devices": [
            {
                "ordinal": ordinal,
                "name": "simulated GPU",
                "total_bytes": 1_000_000,
                "free_bytes": 900000,
                "compute_capability": [8, 0],
            }
            for ordinal in range(2)
        ],
    }


def test_analytical_fit_without_probe_remains_blocked():
    report = resource_report()
    admission = admit_gpu_resources(report, GpuTopology((0, 1)))
    assert all(device["fits_current_observation"] for device in admission["devices"])
    assert admission["preflight_ready"] is False
    assert admission["blockers"] == ["host_not_probed"]
    assert admission["training_executed"] is False
    assert admission["tokens_per_optimizer_step"] == 128


def test_observed_memory_is_checked_on_every_requested_device():
    probe = simulated_probe()
    passed = admit_gpu_resources(resource_report(), GpuTopology((0, 1)), probe)
    assert passed["preflight_ready"] is True
    probe["devices"][1]["free_bytes"] = 0
    blocked = admit_gpu_resources(resource_report(), GpuTopology((0, 1)), probe)
    assert blocked["preflight_ready"] is False
    assert "device_1_memory_budget_exceeded" in blocked["blockers"]


def test_native_bfloat16_and_visible_mapping_are_admission_requirements():
    probe = simulated_probe()
    probe["devices"][0]["compute_capability"] = [7, 5]
    admission = admit_gpu_resources(resource_report(), GpuTopology((0, 2)), probe)
    assert "device_0_lacks_native_bfloat16" in admission["blockers"]
    assert "device_2_not_visible" in admission["blockers"]


def test_mismatched_geometry_or_forged_accounting_is_rejected():
    report = resource_report()
    with pytest.raises(ValueError, match="parallel size"):
        admit_gpu_resources(report, GpuTopology((0,)))
    with pytest.raises(ValueError, match="dtype"):
        admit_gpu_resources(report, GpuTopology((0, 1), dtype="float32"))
    forged = deepcopy(report)
    forged["memory"]["per_device"]["total"] = 1
    with pytest.raises(ValueError, match="recomputation"):
        admit_gpu_resources(forged, GpuTopology((0, 1)))
