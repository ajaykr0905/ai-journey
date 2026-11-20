import dataclasses
import itertools
import os
import sys
import time
import warnings

import pytest

from ai_journey.gpu_ceiling_source import GpuCeilingError
from ai_journey.gpu_reduce_ceiling import load_ceiling
from ai_journey.gpu_reduce_config import config_from_mapping
from ai_journey.gpu_reduce_plan import plan_mapping, plan_sha256
from ai_journey.gpu_reduce_report import (
    MEASUREMENT_BOUNDARY,
    SYNTHETIC_BOUNDARY,
    measure_report,
)
from ai_journey.gpu_reduce_runtime import SIDES
from ai_journey.gpu_reduce_source import (
    KERNELS,
    R2,
    R3,
    STEP_KERNELS,
    GpuReduceError,
    library_sha256,
    library_source,
)
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.matmul_tiled_report import MAX_WARNING_TEXT
from ai_journey.memory_access_report import report_digest
from test_gpu_reduce_config import make_config_mapping
from test_gpu_reduce_correctness import StandInBackend
from test_gpu_reduce_runtime import fake_runtime, reduce_runtime
from test_gpu_reduce_speedup import write_ceiling
from test_kernel_metal_runtime import requires_metal

LOAD = (1.5, 2.25, 3.0)


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


@pytest.fixture
def ceiling(tmp_path):
    return load_ceiling(write_ceiling(tmp_path))


def stand_in_report(ceiling, backend=None, mapping=None, clock=None):
    return measure_report(
        config_from_mapping(mapping or make_config_mapping()),
        ceiling,
        backend=backend or StandInBackend(),
        clock=clock or scripted_clock(),
        load_average=lambda: LOAD,
    )


class WarningBackend(StandInBackend):
    """A stand-in whose runs raise warnings, as PyTorch can on first use."""

    def run(self, name, values):
        home = os.path.expanduser("~")
        warnings.warn(f"{home}/lib/mps.py: fell back to MPSGraph", UserWarning)
        if name == "reduce_r0_interleaved":
            warnings.warn("x" * (MAX_WARNING_TEXT + 50), DeprecationWarning)
        return super().run(name, values)


def test_a_stand_in_report_is_synthetic_sealed_and_complete(ceiling):
    torch_imported = sys.modules.get("torch") is not None

    report = stand_in_report(ceiling)

    assert report["report_sha256"] == report_digest(report)
    assert (report["measured"], report["backend"], report["timer"]) == (
        False,
        "injected",
        "injected",
    )
    assert report["measurement_scope"] == "synthetic_backend_or_clock"
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert report["plan"] == {**plan_mapping(), "sha256": plan_sha256()}
    assert report["host_load"]["before"] == report["host_load"]["after"] == list(LOAD)
    assert report["warnings"] == {
        "capture": report["warnings"]["capture"],
        "torch_imported_before_capture": torch_imported,
        "count": 0,
        "distinct": 0,
        "records": [],
    }
    points = report["benchmark"]["points"]
    assert [point["n"] for point in points] == [4096, 70001]
    assert [list(point["sides"]) for point in points] == [list(SIDES)] * 2
    summary = report["summary"]
    assert summary["timed_samples"] == 2 * 7 * 3
    assert summary["kernels_checked"] == list(STEP_KERNELS)
    assert summary["exact_cases"] == [0, 4, 5]
    assert summary["same_order"] == [
        {
            "kernel": R3,
            "same_order_as": R2,
            "cases_with_other_bits": [],
            "points_with_other_bits": [],
        }
    ]
    assert set(summary["hypotheses"]) == {f"R{index}" for index in range(6)}
    assert summary["fraction_measured"] is False
    assert summary["ceiling_report_sha256"] == ceiling.report_sha256
    assert report["run"]["timed_ns"] <= report["run"]["elapsed_ns"]
    assert MEASUREMENT_BOUNDARY["warp_level_claim"] is False
    assert MEASUREMENT_BOUNDARY["hardware_counters_used"] is False


def test_the_ceiling_section_and_every_kernel_digest_and_limit_are_recorded(ceiling):
    report = stand_in_report(ceiling)

    section = report["ceiling"]
    assert section["report_sha256"] == ceiling.report_sha256
    assert section["gb_per_second"] == ceiling.gb_per_second
    assert section["same_environment"] == ceiling.same_environment(
        report["environment"]
    )
    shaders = report["shaders"]
    assert shaders["library_sha256"] == library_sha256()
    assert [entry["name"] for entry in shaders["kernels"]] == list(KERNELS)
    for entry in shaders["kernels"]:
        assert entry["source_sha256"] == KERNELS[entry["name"]].source_sha256
        assert entry["admission"]["admitted"] is True
    combine = report["correctness"]["combine_launch"]
    assert combine["groups"] == [1] and combine["threads_per_group"] == 1024
    assert report["speedup"]["ceiling_gb_per_second"] == ceiling.gb_per_second


def test_warnings_from_the_run_are_sealed_once_each_without_the_home_path(ceiling):
    report = stand_in_report(ceiling, WarningBackend())

    captured = report["warnings"]
    runs = 6 * 6 * 2
    assert captured["count"] == runs + 6 * 2
    assert captured["distinct"] == 2 and len(captured["records"]) == 2
    fallback, long = captured["records"]
    assert fallback["category"] == "UserWarning" and fallback["count"] == runs
    assert fallback["message"] == "~/lib/mps.py: fell back to MPSGraph"
    assert fallback["file"] == "test_gpu_reduce_report.py"
    assert os.path.expanduser("~") not in str(captured)
    assert long["category"] == "DeprecationWarning" and len(long["message"]) == (
        MAX_WARNING_TEXT
    )
    assert report["summary"]["warnings"] == runs + 12


def test_the_mps_switches_come_from_the_backend_and_bad_loads_are_refused(ceiling):
    backend = StandInBackend()
    backend.mps_switches = {
        "PYTORCH_MPS_FAST_MATH": "1",
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    assert stand_in_report(ceiling, backend)["torch_mps_switches"]["values"] == {
        "PYTORCH_MPS_FAST_MATH": "1",
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    with pytest.raises(GpuCeilingError, match="three nonnegative floats"):
        measure_report(
            config_from_mapping(make_config_mapping()),
            ceiling,
            backend=StandInBackend(),
            clock=scripted_clock(),
            load_average=lambda: (1.0, 2.0),
        )


def test_a_real_runtime_refuses_another_library_or_a_foreign_ceiling(ceiling):
    runtime = fake_runtime()
    config = config_from_mapping(make_config_mapping())
    runtime.library_source = library_source().replace("256u", "128u", 1)
    with pytest.raises(GpuReduceError, match="another library"):
        measure_report(config, ceiling, backend=runtime, clock=scripted_clock())
    runtime.library_source = library_source()
    foreign = dataclasses.replace(
        ceiling, environment={**ceiling.environment, "torch_version": "2.14.0"}
    )
    with pytest.raises(GpuReduceError, match="from the same interpreter"):
        measure_report(config, foreign, backend=runtime, clock=time.perf_counter_ns)
    with pytest.raises(TypeError, match="BandwidthCeiling"):
        measure_report(config, None, backend=StandInBackend())


def test_a_run_past_its_budget_stops_without_a_report(ceiling):
    mapping = make_config_mapping()
    mapping["benchmark"]["max_runtime_seconds"] = 1

    with pytest.raises(KernelMetalError, match="exceeded max_runtime_seconds=1"):
        stand_in_report(ceiling, mapping=mapping, clock=scripted_clock(step=10**9))


@requires_metal
def test_a_real_run_is_a_labeled_apple_gpu_measurement(ceiling):
    report = measure_report(
        config_from_mapping(make_config_mapping()), ceiling, backend=reduce_runtime()
    )

    assert report["measured"] is True
    assert report["label"] == "host-specific Apple GPU (Metal) measurement"
    assert report["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert report["backend"] == "torch.mps.compile_shader"
    assert report["warnings"]["torch_imported_before_capture"] is True
    assert report["summary"]["fraction_measured"] is True
    for case in report["correctness"]["cases"]:
        for output in case["outputs"]:
            assert output["bound_ratio"] <= 1.0
