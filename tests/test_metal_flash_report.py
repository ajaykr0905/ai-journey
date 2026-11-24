import dataclasses
import json
import time

import pytest

# The report records whether torch was imported before its warning capture.
# Import it here, so the answer is the same on macOS and on the Linux CI job.
import torch  # noqa: F401

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.memory_access_report import DIGEST_FIELD
from ai_journey.metal_flash_ceiling import FlashCeiling, load_flash_ceiling
from ai_journey.metal_flash_config import config_from_mapping
from ai_journey.metal_flash_plan import PLAN, SIDES, plan_sha256
from ai_journey.metal_flash_report import (
    GPU_ONLY,
    MEASUREMENT_BOUNDARY,
    SYNTHETIC_BOUNDARY,
    build_report,
    measure_report,
)
from ai_journey.metal_flash_runtime import MetalFlashRuntime
from ai_journey.metal_flash_source import MetalFlashError
from test_gpu_ceiling_recorded import RECORDED
from test_kernel_metal_runtime import requires_metal
from test_metal_flash_emulator import FlashEmulator
from test_metal_fused_ceiling import host_drift, write_recorded
from test_metal_liger_emulator import ticking_clock

SMALL_CONFIG = {
    "schema_version": 1,
    "name": "metal-flash-small",
    "seed": 81,
    "plan_sha256": plan_sha256(),
    "correctness": {
        "cases": [
            {"lengths": [1, 31, 33, 100], "inputs": "normal"},
            {"lengths": [1000, 40], "inputs": "wide_scores"},
        ],
        "repeats": 2,
    },
    "benchmark": {
        "inputs": "normal",
        "shapes": [{"sequences": 2, "length": 64}],
        "modes": ["full", "causal"],
        "check_rows": 8,
        "warmup": 1,
        "repeats": 3,
        "target_bytes_per_sample": 1 << 20,
        "max_runtime_seconds": 60,
    },
    "comparison": {"min_faster_ratio": 1.1},
}


def small_config(**changes):
    payload = json.loads(json.dumps(SMALL_CONFIG))
    for section, values in changes.items():
        payload[section].update(values)
    return config_from_mapping(payload)


def recorded_flash_ceiling(directory):
    return load_flash_ceiling(write_recorded(directory))


def synthetic_report(directory, *, config=None, backend=None, clock=None):
    return measure_report(
        config or small_config(),
        recorded_flash_ceiling(directory),
        backend=backend or FlashEmulator(),
        clock=clock or ticking_clock(),
        load_average=lambda: (1.0, 2.0, 3.0),
    )


def test_a_stand_in_backend_makes_a_sealed_synthetic_report(tmp_path):
    report = synthetic_report(tmp_path)

    assert report["measured"] is False
    assert report["backend"] == "injected"
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert report["evidence_boundary"]["ptx_executed"] is False
    assert len(report[DIGEST_FIELD]) == 64
    assert report["plan"]["sha256"] == plan_sha256()
    assert report["warnings"]["torch_imported_before_capture"] is True
    summary = report["summary"]
    assert summary["correctness_cases"] == 2
    assert summary["benchmark_points"] == 2
    assert summary["unscaled_exp_overflow_cases"] == [1]
    assert set(summary["hypotheses"]) == {step.step for step in PLAN}
    assert summary["load_average_before"] == [1.0, 2.0, 3.0]
    assert set(summary["points"][0]["median_ms_per_call"]) == set(SIDES)
    assert any("allocator reading" in text for text in GPU_ONLY)


def test_both_ceilings_come_from_one_verified_report(tmp_path):
    report = synthetic_report(tmp_path)
    summary = RECORDED["summary"]

    assert (
        report["ceiling"]["gb_per_second"]
        == (summary["bandwidth_ceiling_gb_per_second"])
    )
    assert (
        report["compute_ceiling"]["fma_gflops_per_second"]
        == (summary["fma_ceiling_gflops_per_second"])
    )
    assert (
        report["compute_ceiling"]["report_sha256"]
        == (report["ceiling"]["report_sha256"])
    )
    row = report["speedup"]["points"][0]["rates"]["f1"]
    assert row["fraction_of_fma"] == pytest.approx(
        row["gflops_per_second"] / summary["fma_ceiling_gflops_per_second"]
    )


def test_a_tampered_ceiling_report_sets_no_ceiling(tmp_path):
    path = write_recorded(tmp_path)
    path.write_text(path.read_text().replace('"27.2"', '"27.3"'), encoding="ascii")

    with pytest.raises(MetalFlashError, match="refused"):
        load_flash_ceiling(path)
    with pytest.raises(MetalFlashError, match="positive"):
        dataclasses.replace(recorded_flash_ceiling(tmp_path), fma_gflops_per_second=0.0)
    with pytest.raises(TypeError):
        FlashCeiling(bandwidth=None, fma_gflops_per_second=1.0)


def test_a_refused_pipeline_stops_the_run_before_any_measurement(tmp_path):
    backend = FlashEmulator(limits={"flash_f1": PipelineLimits(1024, 32, 8192)})

    with pytest.raises(MetalFlashError, match="refused flash_f1"):
        synthetic_report(tmp_path, backend=backend)


def test_a_run_past_its_budget_stops_without_a_report(tmp_path):
    config = small_config(benchmark={"max_runtime_seconds": 1})

    with pytest.raises(ValueError, match="max_runtime_seconds"):
        synthetic_report(tmp_path, config=config, clock=ticking_clock(step=10**8))


def test_a_measured_run_needs_a_measured_ceiling(tmp_path):
    from ai_journey.metal_flash_report import _require_ceiling

    ceiling = recorded_flash_ceiling(tmp_path)
    host = dict(ceiling.bandwidth.environment)
    _require_ceiling(ceiling, True, host)
    synthetic = dataclasses.replace(
        ceiling,
        bandwidth=dataclasses.replace(
            ceiling.bandwidth,
            measured=False,
            label="synthetic timing, not a measurement",
        ),
    )
    with pytest.raises(MetalFlashError, match="measured ceiling"):
        _require_ceiling(synthetic, True, host)
    with pytest.raises(MetalFlashError, match="another interpreter"):
        _require_ceiling(ceiling, True, {**host, "numpy_version": "2.0.0"})
    with pytest.raises(TypeError):
        build_report(small_config(), ceiling.bandwidth)


@requires_metal
def test_this_gpu_makes_a_measured_report_with_a_ceiling_of_this_host(tmp_path):
    ceiling = recorded_flash_ceiling(tmp_path)
    drift = host_drift(ceiling.bandwidth)
    if drift:
        with pytest.raises(MetalFlashError, match="another interpreter"):
            measure_report(small_config(), ceiling, clock=time.perf_counter_ns)
        pytest.skip(f"the recorded ceiling comes from another host: {drift}")
    report = measure_report(small_config(), ceiling, clock=time.perf_counter_ns)

    assert report["measured"] is True
    assert report["evidence_boundary"] == MEASUREMENT_BOUNDARY


@requires_metal
def test_this_gpu_with_an_injected_clock_makes_only_a_synthetic_report(tmp_path):
    report = measure_report(
        small_config(),
        recorded_flash_ceiling(tmp_path),
        backend=MetalFlashRuntime(),
        clock=ticking_clock(),
    )

    assert report["measured"] is False
    assert report["backend"] == "torch.mps.compile_shader"
    assert all(entry["admitted"] for entry in report["shaders"]["admissions"])
    assert max(report["summary"]["max_bound_ratio"].values()) <= 1
