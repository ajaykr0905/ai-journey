import copy
import dataclasses
import json
import time
from pathlib import Path

import pytest
import torch

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.metal_tc_ceiling import load_tc_ceiling
from ai_journey.metal_tc_config import config_from_mapping
from ai_journey.metal_tc_plan import SIDES, plan_mapping, plan_sha256
from ai_journey.metal_tc_report import (
    MEASUREMENT_BOUNDARY,
    SYNTHETIC_BOUNDARY,
    _require_ceiling,
    build_report,
    measure_report,
)
from ai_journey.metal_tc_source import KERNEL_NAME, M1, M3, MetalTcError
from test_kernel_metal_runtime import requires_metal
from test_metal_fused_ceiling import write_recorded
from test_metal_liger_emulator import ticking_clock
from test_metal_tc_emulator import TcEmulator

ROOT = Path(__file__).resolve().parents[1]
TORCH_IMPORTED = torch is not None


def small_config(**changes):
    payload = json.loads((ROOT / "config" / "day-82-metal-tc.json").read_text())
    payload["name"] = "metal-tc-small"
    payload["correctness"]["cases"] = [
        {"m": 1, "n": 1, "k": 1},
        {"m": 7, "n": 9, "k": 5},
        {"m": 33, "n": 65, "k": 17},
        {"m": 64, "n": 32, "k": 32},
        {"m": 34, "n": 41, "k": 1000},
    ]
    payload["benchmark"].update(
        {
            "sizes": [32, 40],
            "check_rows": 4,
            "warmup": 1,
            "repeats": 7,
            "target_flops_per_sample": 1 << 20,
            "max_runtime_seconds": 120,
        }
    )
    for section, values in changes.items():
        payload[section].update(values)
    return config_from_mapping(payload)


def synthetic_report(directory, backend=None, config=None, clock=None):
    return measure_report(
        config or small_config(),
        load_tc_ceiling(write_recorded(directory)),
        backend=backend or TcEmulator(),
        clock=clock or ticking_clock(),
        load_average=lambda: (1.0, 2.0, 3.0),
    )


def test_a_stand_in_run_is_sealed_as_synthetic_with_every_section(tmp_path):
    report = synthetic_report(tmp_path)

    assert report["measured"] is False
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert report["plan"] == {**plan_mapping(), "sha256": plan_sha256()}
    assert report["support"]["metal_language_version"] == 400
    assert report["warnings"]["torch_imported_before_capture"] is TORCH_IMPORTED
    assert [point["size"] for point in report["benchmark"]["points"]] == [32, 40]
    summary = report["summary"]
    assert summary["correctness_cases"] == 5
    assert summary["timed_samples"] == 2 * 7 * len(SIDES)
    assert summary["load_average_before"] == [1.0, 2.0, 3.0]
    assert summary["m1_same_bits_as_s1_cases"] == 5
    assert set(summary["hypotheses"]) == {
        step["step"] for step in report["plan"]["steps"]
    }
    assert len(report["report_sha256"]) == 64


def test_an_unsupported_bfloat_path_is_recorded_and_judged_nowhere(tmp_path):
    report = synthetic_report(tmp_path, backend=TcEmulator(bfloat=False))
    summary = report["summary"]

    assert summary["supported"] == {"m3": False, "torch_bfloat16": True}
    assert M3 not in summary["max_bound_ratio"]
    assert summary["hypotheses"]["m3_over_s1"] == {
        "held": 0,
        "missed": 0,
        "not_run": 2,
    }
    names = [entry["kernel"] for entry in report["shaders"]["admissions"]]
    assert KERNEL_NAME[M3] not in names
    limits = {e["name"]: e["limits"] for e in report["shaders"]["kernels"]}
    assert limits[KERNEL_NAME[M3]] is None


def test_the_boundary_names_tensor_cores_as_absent():
    for boundary in (MEASUREMENT_BOUNDARY, SYNTHETIC_BOUNDARY):
        assert boundary["tensor_cores_used"] is False
        assert boundary["nvidia_gpu_used"] is False
    assert "Apple GPUs have no NVIDIA Tensor Cores" in MEASUREMENT_BOUNDARY["statement"]


def test_a_refused_pipeline_stops_the_run_before_any_measurement(tmp_path):
    backend = TcEmulator(limits={KERNEL_NAME[M1]: PipelineLimits(1024, 32, 64)})

    with pytest.raises(MetalTcError, match="refused tc_m1_float"):
        synthetic_report(tmp_path, backend=backend)


def test_a_run_past_its_budget_stops_without_a_report(tmp_path):
    config = small_config(benchmark={"max_runtime_seconds": 1})

    with pytest.raises(ValueError, match="max_runtime_seconds"):
        synthetic_report(tmp_path, config=config, clock=ticking_clock(step=10**8))


def test_a_measured_run_needs_a_measured_ceiling_of_this_environment(tmp_path):
    ceiling = load_tc_ceiling(write_recorded(tmp_path))
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
    with pytest.raises(MetalTcError, match="measured ceiling"):
        _require_ceiling(synthetic, True, host)
    with pytest.raises(MetalTcError, match="another interpreter"):
        _require_ceiling(ceiling, True, {**host, "numpy_version": "2.0.0"})
    with pytest.raises(TypeError):
        build_report(small_config(), ceiling.bandwidth)


def test_the_report_keeps_each_raw_sample_beside_its_rates(tmp_path):
    report = synthetic_report(tmp_path)
    point = copy.deepcopy(report["benchmark"]["points"][1])

    for side, entry in point["sides"].items():
        assert len(entry["samples_ns"]) == 7, side
        assert entry["tflops_per_second"] > 0
    row = report["speedup"]["points"][1]
    assert set(row["rates"]) == set(point["sides"])
    assert {step["step"] for step in row["steps"]} == set(
        report["summary"]["hypotheses"]
    )


@requires_metal
def test_this_gpu_refuses_a_ceiling_from_another_environment(tmp_path):
    ceiling = load_tc_ceiling(write_recorded(tmp_path))
    environment = {**ceiling.bandwidth.environment, "numpy_version": "0.0.0"}
    foreign = dataclasses.replace(
        ceiling,
        bandwidth=dataclasses.replace(ceiling.bandwidth, environment=environment),
    )

    with pytest.raises(MetalTcError, match="another interpreter"):
        measure_report(small_config(), foreign, clock=time.perf_counter_ns)
