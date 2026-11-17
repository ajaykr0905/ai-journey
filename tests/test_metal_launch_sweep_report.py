import itertools
import math
import time

import pytest

from ai_journey import metal_launch_sweep_report
from ai_journey.kernel_metal_geometry import KernelMetalError, PipelineLimits
from ai_journey.kernel_metal_source import source_sha256 as day_73_source_sha256
from ai_journey.memory_access_report import report_digest
from ai_journey.metal_launch_sweep_config import config_from_mapping
from ai_journey.metal_launch_sweep_report import (
    MEASUREMENT_BOUNDARY,
    SYNTHETIC_BOUNDARY,
    load_average,
    measure_report,
)
from ai_journey.metal_launch_sweep_shapes import MetalLaunchSweepError
from ai_journey.metal_launch_sweep_source import source_sha256
from test_kernel_metal_runtime import requires_metal
from test_metal_launch_sweep_config import make_config_mapping
from test_metal_launch_sweep_standin import StandInBackend


def scripted_clock():
    return itertools.count(0, 1_000).__next__


def stand_in_report(**options):
    config = config_from_mapping(make_config_mapping())
    clock = options.pop("clock", None) or scripted_clock()
    return measure_report(config, backend=StandInBackend(**options), clock=clock)


def test_a_stand_in_run_is_sealed_and_labeled_synthetic():
    report = stand_in_report()

    assert report["report_sha256"] == report_digest(report)
    assert (report["measured"], report["backend"], report["timer"]) == (
        False,
        "injected",
        "injected",
    )
    assert report["measurement_scope"] == "synthetic_backend_or_clock"
    assert report["label"] == "synthetic timing, not a measurement"
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert report["config"] == make_config_mapping()
    assert report["shaders"]["source_sha256"] == source_sha256()
    assert report["shaders"]["day_73_source_sha256"] == day_73_source_sha256()
    assert report["shaders"]["day_73_kernel"]["bytes_per_item"] == 12
    assert set(report["pipelines"]) == {
        "matmul_naive",
        "vector_add",
        "simd_layout_probe",
    }
    assert report["dispatch"]["requires_whole_simd_groups"] is False
    assert report["mapping"]["case"] == [13, 19, 3]
    assert len(report["mapping"]["checks"]) == 3
    assert len(report["simd_layout"]) == 3 + 4


def test_the_report_holds_every_check_point_and_run_total():
    report = stand_in_report()

    correctness = report["correctness"]
    assert [case["index"] for case in correctness["matmul"]] == [0, 1]
    assert [case["index"] for case in correctness["vector_add"]] == [0, 1, 2, 3]
    benchmark = report["benchmark"]
    assert [point["m"] for point in benchmark["matmul"]] == [32, 64]
    assert [point["length"] for point in benchmark["vector"]] == [4096, 65537]
    run = report["run"]
    samples = [
        sample
        for point in (*benchmark["matmul"], *benchmark["vector"])
        for shape in point["shapes"]
        for sample in shape["samples_ns"]
    ]
    assert run["timed_ns"] == sum(samples)
    assert run["timed_ns"] <= run["elapsed_ns"] <= run["max_runtime_ns"]
    for key in ("load_average_before", "load_average_after"):
        loads = run[key]
        assert loads is None or (
            len(loads) == 3 and all(math.isfinite(value) for value in loads)
        )
    summary = report["summary"]
    assert (summary["admitted_shapes"], summary["rejected_shapes"]) == (7, 0)
    assert (summary["correctness_cases"], summary["benchmark_points"]) == (6, 4)
    assert summary["timed_samples"] == len(samples) == 3 * (3 * 2 + 4 * 2)
    assert 0 < summary["matmul_max_bound_ratio"] < 1
    assert [row["m"] for row in summary["matmul"]] == [32, 64]
    assert summary["best_vector_gb_per_second"] == max(
        row["best_median_gb_per_second"] for row in summary["vector"]
    )
    assert report["planned_work"]["timed_samples"] == summary["timed_samples"]


def test_rejected_shapes_are_recorded_and_never_launched():
    report = stand_in_report(
        limits=PipelineLimits(128, 32, 0), requires_whole_simd_groups=True
    )

    matmul = report["admission"]["matmul"]
    vector = report["admission"]["vector"]
    assert [entry["reason"] for entry in matmul] == [
        None,
        "exceeds_max_threads_per_threadgroup",
        "exceeds_max_threads_per_threadgroup",
    ]
    assert [entry["reason"] for entry in vector] == [
        "not_whole_simd_groups",
        None,
        "not_whole_simd_groups",
        "exceeds_max_threads_per_threadgroup",
    ]
    assert report["dispatch"]["requires_whole_simd_groups"] is True
    assert [check["group_size"] for check in report["mapping"]["checks"]] == [[8, 8]]
    assert [item["group_size"] for item in report["simd_layout"]] == [[8, 8], [32]]
    assert [case["index"] for case in report["correctness"]["vector_add"]] == [1]
    for point in report["benchmark"]["matmul"]:
        assert point["ranking"] == [[8, 8]] and point["spread"] == 1.0
    summary = report["summary"]
    assert (summary["admitted_shapes"], summary["rejected_shapes"]) == (2, 5)
    assert summary["timed_samples"] == 3 * (1 * 2 + 1 * 2)


def test_only_the_real_runtime_and_the_host_timer_make_a_measurement():
    report = stand_in_report(clock=time.perf_counter_ns)

    assert report["timer"] == "time.perf_counter_ns"
    assert (report["measured"], report["backend"]) == (False, "injected")
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY


def test_a_run_over_budget_or_with_an_unstated_rule_stops():
    mapping = make_config_mapping()
    mapping["timing"]["max_runtime_seconds"] = 1
    config = config_from_mapping(mapping)
    clock = itertools.count(0, 10**8).__next__

    with pytest.raises(KernelMetalError, match="max_runtime_seconds=1"):
        measure_report(config, backend=StandInBackend(), clock=clock)
    backend = StandInBackend()
    backend.requires_whole_simd_groups = 0
    with pytest.raises(MetalLaunchSweepError, match="SIMD rule as a bool"):
        measure_report(config, backend=backend, clock=scripted_clock())


def test_load_averages_are_rounded_or_absent(monkeypatch):
    monkeypatch.setattr(
        metal_launch_sweep_report.os, "getloadavg", lambda: (1.234, 5.0, 9.876)
    )
    assert load_average() == [1.23, 5.0, 9.88]

    def unavailable():
        raise OSError("no load average")

    monkeypatch.setattr(metal_launch_sweep_report.os, "getloadavg", unavailable)
    assert load_average() is None
    monkeypatch.setattr(
        metal_launch_sweep_report.os, "getloadavg", lambda: (1.0, math.nan, 2.0)
    )
    assert load_average() is None


def test_the_measurement_boundary_denies_every_claim_beyond_this_host():
    for key in (
        "cuda_kernel_executed",
        "nvidia_gpu_used",
        "distributed_claim",
        "production_claim",
        "portable_performance_claim",
        "vendor_peak_claim",
        "scheduler_internals_claim",
    ):
        assert MEASUREMENT_BOUNDARY[key] is False, key
        assert SYNTHETIC_BOUNDARY[key] is False, key
    statement = MEASUREMENT_BOUNDARY["statement"]
    assert "host-specific Apple GPU (Metal) measurements" in statement
    assert "not a vendor peak" in statement
    assert "not how the GPU schedules SIMD groups" in statement


@requires_metal
def test_a_small_run_on_this_gpu_is_a_labeled_measurement():
    config = config_from_mapping(make_config_mapping())

    report = measure_report(config)

    assert (report["measured"], report["backend"], report["timer"]) == (
        True,
        "torch.mps.compile_shader",
        "time.perf_counter_ns",
    )
    assert report["label"] == "host-specific Apple GPU (Metal) measurement"
    assert report["measurement_scope"] == "apple_gpu_metal_this_host"
    assert report["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert report["environment"]["platform_system"] == "Darwin"
    assert report["summary"]["rejected_shapes"] == 0
