import dataclasses
import time
import warnings

import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.memory_access_report import report_digest
from ai_journey.metal_fused_config import config_from_mapping
from ai_journey.metal_fused_plan import TORCH_SOFTMAX, plan_mapping, plan_sha256
from ai_journey.metal_fused_report import (
    MEASUREMENT_BOUNDARY,
    REPORT_KIND,
    SYNTHETIC_BOUNDARY,
    _require_ceiling,
    build_report,
    measure_report,
)
from ai_journey.metal_fused_source import (
    F1,
    F3,
    KERNELS,
    STEP_VARIANTS,
    U0,
    MetalFusedError,
    library_sha256,
)
from test_kernel_metal_runtime import requires_metal
from test_metal_fused_benchmark import StepClock
from test_metal_fused_ceiling import host_drift, recorded_ceiling
from test_metal_fused_config import make_config_mapping
from test_metal_fused_correctness import StandInBackend, recorded_limits

LOAD = (2.5, 2.0, 1.5)


def small_config(**sections):
    return config_from_mapping(make_config_mapping(**sections))


def synthetic_report(tmp_path, backend=None, config=None, clock=None):
    return measure_report(
        config or small_config(),
        recorded_ceiling(tmp_path),
        backend=backend or StandInBackend(),
        clock=clock or StepClock(),
        load_average=lambda: LOAD,
    )


def test_a_stand_in_run_seals_a_synthetic_report(tmp_path):
    report = synthetic_report(tmp_path)

    assert report["kind"] == REPORT_KIND
    assert report["measured"] is False
    assert report["backend"] == "injected" and report["timer"] == "injected"
    assert report["label"] == "synthetic timing, not a measurement"
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert report["report_sha256"] == report_digest(report)
    assert report["plan"] == {**plan_mapping(), "sha256": plan_sha256()}
    assert report["config"] == small_config().to_mapping()
    assert report["shaders"]["library_sha256"] == library_sha256()
    assert report["refusals"] == []
    assert report["host_load"]["before"] == list(LOAD)
    assert (
        report["ceiling"]["gb_per_second"] == recorded_ceiling(tmp_path).gb_per_second
    )
    assert report["exp_probe"]["holds"] is True
    assert len(report["correctness"]["cases"]) == 2
    assert len(report["benchmark"]["points"]) == 2
    assert report["run"]["timed_ns"] <= report["run"]["elapsed_ns"]


def test_the_summary_reconciles_with_the_sections(tmp_path):
    report = synthetic_report(tmp_path)
    summary = report["summary"]

    assert summary["steps_checked"] == list(STEP_VARIANTS)
    assert summary["refused_steps"] == []
    assert summary["correctness_cases"] == 2
    assert summary["naive_float32_failed_cases"] == [1]
    assert summary["same_bits"] == [
        {"variant": F1, "as": U0, "same_at": [0, 1], "differ_at": []}
    ]
    assert summary["identical_repeats"] == dict.fromkeys(STEP_VARIANTS, True)
    assert summary["timed_samples"] == 2 * 5 * 5
    assert summary["points"][0]["shape"] == {"rows": 8, "cols": 64}
    assert set(summary["points"][0]["effective_gb_per_second"]) == {
        *STEP_VARIANTS,
        TORCH_SOFTMAX,
    }
    assert set(summary["hypotheses"]) == set(STEP_VARIANTS)
    assert summary["load_average_after"] == list(LOAD)
    assert summary["ceiling_gb_per_second"] == report["ceiling"]["gb_per_second"]


def test_a_refused_step_is_recorded_and_never_runs(tmp_path):
    limits = recorded_limits(**{F3: PipelineLimits(512, 32, 256)})
    config = small_config(
        correctness={
            "cases": [
                {"lengths": [1, 31, 33, 9000], "inputs": "uniform_80"},
                {"rows": 2, "cols": 64, "inputs": "offset_plus_1000"},
            ]
        }
    )
    backend = StandInBackend(limits=limits)

    report = synthetic_report(tmp_path, backend=backend, config=config)

    assert config.largest_group() == 1024
    assert [entry["variant"] for entry in report["refusals"]] == [F3]
    assert report["refusals"][0]["reason"] == (
        "pipeline_allows_fewer_threads_than_the_group"
    )
    assert F3 not in {name for name, _ in backend.runs}
    assert report["summary"]["refused_steps"] == [F3]
    assert F3 not in report["benchmark"]["points"][0]["sides"]
    assert report["summary"]["hypotheses"][F3] == {
        "held": 0,
        "missed": 0,
        "no_verdict": 2,
    }


def test_a_run_with_every_step_refused_stops_before_it_measures(tmp_path):
    narrow = {
        name: PipelineLimits(1024, 16, KERNELS[name].threadgroup_bytes)
        for name in ("softmax_u0_row_max", F1, "softmax_f2_simd", F3)
    }
    backend = StandInBackend(limits=recorded_limits(**narrow))

    with pytest.raises(MetalFusedError, match="every step was refused"):
        synthetic_report(tmp_path, backend=backend)
    assert backend.runs == []


def test_the_capture_seals_each_warning_of_the_run(tmp_path):
    class Warns(StandInBackend):
        def exp_probe(self, arguments):
            warnings.warn("a one-time MPS notice", UserWarning, stacklevel=1)
            return super().exp_probe(arguments)

    report = synthetic_report(tmp_path, backend=Warns())

    captured = report["warnings"]
    assert captured["count"] == 1 and captured["distinct"] == 1
    assert captured["records"][0]["message"] == "a one-time MPS notice"
    assert type(captured["torch_imported_before_capture"]) is bool
    assert report["summary"]["warnings"] == 1
    assert report["torch_mps_switches"]["values"] == {
        "PYTORCH_MPS_FAST_MATH": None,
        "PYTORCH_MPS_PREFER_METAL": None,
    }


def test_a_measured_run_needs_a_measured_ceiling_of_its_environment(tmp_path):
    ceiling = recorded_ceiling(tmp_path)
    host = dict(ceiling.environment)

    _require_ceiling(ceiling, True, host)
    synthetic = dataclasses.replace(
        ceiling, measured=False, label="synthetic timing, not a measurement"
    )
    with pytest.raises(MetalFusedError, match="measured ceiling"):
        _require_ceiling(synthetic, True, host)
    _require_ceiling(synthetic, False, host)
    with pytest.raises(MetalFusedError, match="another interpreter"):
        _require_ceiling(ceiling, True, {**host, "numpy_version": "2.0.0"})
    with pytest.raises(TypeError):
        build_report(small_config(), {"gb_per_second": 80.0})


def test_a_run_past_its_budget_stops_without_a_report(tmp_path):
    config = small_config(benchmark={"max_runtime_seconds": 1})

    with pytest.raises(ValueError, match="max_runtime_seconds"):
        synthetic_report(tmp_path, config=config, clock=StepClock(step=10**8))


@requires_metal
def test_this_gpu_makes_a_measured_report_with_the_recorded_ceiling(tmp_path):
    ceiling = recorded_ceiling(tmp_path)
    drift = host_drift(ceiling)
    if drift:
        with pytest.raises(MetalFusedError, match="another interpreter"):
            measure_report(small_config(), ceiling, clock=time.perf_counter_ns)
        pytest.skip(f"the recorded ceiling comes from another host: {drift}")
    report = measure_report(small_config(), ceiling, clock=time.perf_counter_ns)

    assert report["measured"] is True
    assert report["backend"] == "torch.mps.compile_shader"
    assert report["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert report["summary"]["max_bound_ratio"] <= 1
    assert all(entry["admitted"] for entry in report["shaders"]["admissions"])
