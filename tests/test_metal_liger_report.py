import copy

import pytest

# The report records whether torch was imported before its warning capture.
# Import it here, so the answer is the same on macOS and on the Linux CI job.
import torch  # noqa: F401

from ai_journey.kernel_metal_geometry import KernelMetalError, PipelineLimits
from ai_journey.memory_access_report import DIGEST_FIELD
from ai_journey.metal_liger_config import config_from_mapping
from ai_journey.metal_liger_plan import plan_sha256
from ai_journey.metal_liger_report import (
    GPU_ONLY,
    MEASUREMENT_BOUNDARY,
    SYNTHETIC_BOUNDARY,
    build_report,
    measure_report,
)
from ai_journey.metal_liger_runtime import MetalLigerRuntime
from ai_journey.metal_liger_source import FAMILIES, MetalLigerError
from test_kernel_metal_runtime import requires_metal
from test_metal_fused_ceiling import recorded_ceiling
from test_metal_liger_emulator import LigerEmulator, ticking_clock

SMALL_CONFIG = {
    "schema_version": 1,
    "name": "metal-liger-small",
    "seed": 80,
    "plan_sha256": plan_sha256(),
    "correctness": {
        "cases": {
            "rmsnorm": [{"lengths": [1, 31, 33, 300], "inputs": "uniform_80"}],
            "swiglu": [{"lengths": [1, 31, 33, 300], "inputs": "offset_minus_1000"}],
            "cross_entropy": [
                {"lengths": [1, 31, 33, 16400], "inputs": "uniform_80"},
                {"rows": 2, "cols": 64, "inputs": "offset_plus_1000"},
            ],
        },
        "repeats": 2,
    },
    "benchmark": {
        "inputs": "normal",
        "shapes": {
            "rmsnorm": [{"rows": 64, "cols": 300}],
            "swiglu": [{"rows": 64, "cols": 300}],
            "cross_entropy": [{"rows": 16, "cols": 2049}],
        },
        "warmup": 1,
        "repeats": 4,
        "target_bytes_per_sample": 1 << 18,
        "max_runtime_seconds": 60,
    },
    "comparison": {"min_faster_ratio": 1.1},
}


def small_config(**changes):
    payload = copy.deepcopy(SMALL_CONFIG)
    payload["benchmark"].update(changes)
    return config_from_mapping(payload)


def loads():
    return (1.5, 1.25, 1.0)


def synthetic_report(tmp_path, backend="emulator", clock=None):
    return measure_report(
        small_config(),
        recorded_ceiling(tmp_path),
        backend=LigerEmulator() if backend == "emulator" else backend,
        clock=clock or ticking_clock(),
        load_average=loads,
    )


def test_a_stand_in_run_seals_a_synthetic_report_of_every_family(tmp_path):
    report = synthetic_report(tmp_path)

    assert report["measured"] is False
    assert report["label"] == "synthetic timing, not a measurement"
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert report["backend"] == "injected" and report["timer"] == "injected"
    assert report["plan"]["sha256"] == plan_sha256()
    assert len(report[DIGEST_FIELD]) == 64
    assert [case["family"] for case in report["correctness"]["cases"]] == [
        "rmsnorm",
        "swiglu",
        "cross_entropy",
        "cross_entropy",
    ]
    assert [point["family"] for point in report["benchmark"]["points"]] == list(
        FAMILIES
    )
    summary = report["summary"]
    assert summary["naive_float32_failed_cases"] == [1]
    assert summary["identical_repeats"] == dict.fromkeys(FAMILIES, True)
    assert summary["load_average_before"] == [1.5, 1.25, 1.0]
    assert summary["timed_samples"] == 3 * 2 * 4
    assert set(summary["probe_max_ulps"]) == {"exp", "rsqrt", "log"}
    assert report["verification"]["gpu_only"] == list(GPU_ONLY)
    assert report["warnings"]["torch_imported_before_capture"] is True


def test_every_claim_of_the_boundary_stays_false():
    for boundary in (MEASUREMENT_BOUNDARY, SYNTHETIC_BOUNDARY):
        claims = {key: value for key, value in boundary.items() if key != "statement"}
        assert claims["cuda_kernel_executed"] is False
        assert claims["liger_kernel_code_executed"] is False
        assert claims["backward_pass_measured"] is False
        assert claims["triton_gpu_kernel_executed"] is False


def test_a_refused_pipeline_stops_the_run_before_any_check(tmp_path):
    backend = LigerEmulator(limits={"liger_rmsnorm": PipelineLimits(1024, 32, 0)})

    with pytest.raises(MetalLigerError, match="refused liger_rmsnorm"):
        synthetic_report(tmp_path, backend=backend)


def test_a_run_past_its_budget_stops_without_a_report(tmp_path):
    with pytest.raises(KernelMetalError, match="max_runtime_seconds"):
        measure_report(
            small_config(max_runtime_seconds=1),
            recorded_ceiling(tmp_path),
            backend=LigerEmulator(),
            clock=ticking_clock(step=10**8),
            load_average=loads,
        )


def test_a_measured_run_refuses_a_ceiling_from_another_environment(tmp_path):
    # A real runtime class with a foreign PyTorch version, timed by the host.
    backend = object.__new__(MetalLigerRuntime)
    backend.__dict__.update(LigerEmulator().__dict__)
    backend.torch_version = "0.0.1"
    with pytest.raises(MetalLigerError, match="another interpreter"):
        build_report(small_config(), recorded_ceiling(tmp_path), backend=backend)


@requires_metal
def test_the_apple_gpu_run_with_an_injected_clock_is_synthetic(tmp_path):
    report = synthetic_report(tmp_path, backend=None)

    assert report["backend"] == "torch.mps.compile_shader"
    assert report["measured"] is False and report["timer"] == "injected"
    assert all(verdict["admitted"] for verdict in report["shaders"]["admissions"])
    assert max(report["summary"]["max_bound_ratio"].values()) <= 1
