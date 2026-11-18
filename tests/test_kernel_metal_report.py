import itertools
import json
import platform

import pytest

from ai_journey.kernel_metal_config import config_from_mapping
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.kernel_metal_report import (
    ENVIRONMENT_FIELDS,
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_LABEL,
    REPORT_DIGEST_FIELD,
    SYNTHETIC_BOUNDARY,
    SYNTHETIC_LABEL,
    build_report,
    host_environment,
    measure_report,
)
from ai_journey.kernel_metal_source import SHADER_SOURCE, source_sha256
from ai_journey.memory_access_report import report_digest
from test_kernel_metal_config import changed, make_config_mapping
from test_kernel_metal_recorded import RecordedBackend
from test_kernel_metal_runtime import metal_runtime, requires_metal

CONFIG = config_from_mapping(make_config_mapping())
TOP_LEVEL = {
    "schema_version",
    "kind",
    "measured",
    "measurement_scope",
    "label",
    "evidence_boundary",
    "backend",
    "timer",
    "percentile_method",
    "config",
    "config_sha256",
    "environment",
    "shaders",
    "planned_work",
    "correctness",
    "benchmark",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


def test_a_recorded_backend_report_is_sealed_and_labeled_synthetic():
    report = measure_report(CONFIG, backend=RecordedBackend(), clock=scripted_clock())

    assert set(report) == TOP_LEVEL
    assert report[REPORT_DIGEST_FIELD] == report_digest(report)
    assert report["kind"] == "kernel_metal_correctness_and_benchmark"
    assert (report["measured"], report["label"]) == (False, SYNTHETIC_LABEL)
    assert report["measurement_scope"] == "synthetic_backend_or_clock"
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert (report["backend"], report["timer"]) == ("injected", "injected")
    assert report["config"] == make_config_mapping()
    assert report["config_sha256"] == CONFIG.fingerprint()
    assert report["planned_work"] == CONFIG.planned_work().to_mapping()
    shaders = report["shaders"]
    assert (shaders["source"], shaders["source_sha256"]) == (
        SHADER_SOURCE,
        source_sha256(),
    )
    assert [kernel["name"] for kernel in shaders["kernels"]] == [
        "vector_add",
        "rgb_to_gray",
    ]
    assert shaders["kernels"][0]["limits"]["max_threads_per_threadgroup"] == 1024
    assert shaders["contraction_probe"]["fused"] is True


def test_the_report_holds_every_case_and_point_with_a_reconciled_summary():
    report = measure_report(CONFIG, backend=RecordedBackend(), clock=scripted_clock())
    correctness, benchmark = report["correctness"], report["benchmark"]

    assert [case["length"] for case in correctness["vector_add"]] == [1, 257, 4096]
    assert [(case["height"], case["width"]) for case in correctness["rgb_to_gray"]] == [
        (1, 1),
        (17, 33),
    ]
    assert correctness["bound_levels"] == 0
    assert correctness["sentinels"] == {"float32_bits": 0x7FC00000, "uint8": 0xA5}
    assert [point["length"] for point in benchmark["vector_add"]] == [4096, 65537]
    assert [point["width"] for point in benchmark["rgb_to_gray"]] == [33, 64]
    assert benchmark["device"] == "mps"
    timed = [
        sample
        for points in (benchmark["vector_add"], benchmark["rgb_to_gray"])
        for point in points
        for side in ("metal_kernel", "torch")
        for sample in point[side]["samples_ns"]
    ]
    assert report["run"]["timed_ns"] == sum(timed) == 24 * 1_000
    assert report["run"]["max_runtime_ns"] == 60 * 10**9
    assert report["run"]["elapsed_ns"] >= report["run"]["timed_ns"]
    summary = report["summary"]
    assert summary["label"] == SYNTHETIC_LABEL
    assert (summary["correctness_cases"], summary["vector_add_exact_cases"]) == (5, 3)
    assert summary["rgb_to_gray_max_abs_difference_levels"] == 0
    assert summary["rgb_to_gray_metal_mismatched_pixels"] == 0
    assert (summary["benchmark_points"], summary["timed_samples"]) == (4, 24)
    assert summary["vector_add_largest"]["length"] == 65537
    assert summary["rgb_to_gray_largest"]["height"] == 64
    assert summary["contraction_fused"] is True


def test_the_all_triples_case_reaches_the_summary_with_no_mismatch():
    config = config_from_mapping(
        changed(("correctness", "rgb_to_gray_all_triples"), True)
    )

    report = build_report(config, backend=RecordedBackend(), clock=scripted_clock())

    last = report["correctness"]["rgb_to_gray"][-1]
    assert (last["pattern"], last["index"]) == ("all_rgb_triples", 2)
    assert report["correctness"]["bound_levels"] == 0
    assert report["summary"]["rgb_to_gray_metal_mismatched_pixels"] == 0
    assert report["summary"]["rgb_to_gray_max_abs_difference_levels"] == 0


def test_the_host_timer_with_a_stand_in_backend_is_still_synthetic():
    report = build_report(CONFIG, backend=RecordedBackend())

    assert report["timer"] == "time.perf_counter_ns"
    assert (report["backend"], report["measured"]) == ("injected", False)
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY


def test_the_environment_names_versions_and_platform_but_no_host():
    environment = host_environment("2.15.0")

    assert tuple(environment) == ENVIRONMENT_FIELDS
    assert environment["torch_version"] == "2.15.0"
    assert host_environment("2.15.0 built on my-laptop")["torch_version"] == "unknown"
    assert host_environment(None)["torch_version"] == "unknown"
    node = platform.node()
    if node:
        assert node not in json.dumps(environment)


def test_runs_stop_on_an_unexpected_probe_or_an_exhausted_budget():
    class OddProbe(RecordedBackend):
        def contraction_probe(self):
            return 1.0

    with pytest.raises(KernelMetalError, match="neither the fused nor"):
        build_report(CONFIG, backend=OddProbe(), clock=scripted_clock())
    config = config_from_mapping(changed(("benchmark", "max_runtime_seconds"), 1))
    with pytest.raises(KernelMetalError, match="max_runtime_seconds=1"):
        build_report(config, backend=RecordedBackend(), clock=scripted_clock(10**8))
    with pytest.raises(TypeError, match="KernelMetalConfig"):
        build_report(make_config_mapping(), backend=RecordedBackend())


def test_both_boundaries_rule_out_cuda_and_nvidia_claims():
    for boundary in (MEASUREMENT_BOUNDARY, SYNTHETIC_BOUNDARY):
        assert boundary["cuda_kernel_executed"] is False
        assert boundary["nvidia_gpu_used"] is False
        assert boundary["production_claim"] is False
        assert ";" not in boundary["statement"]
    assert MEASUREMENT_BOUNDARY["apple_gpu_metal_this_host"] is True
    assert "host-specific Apple GPU (Metal)" in MEASUREMENT_BOUNDARY["statement"]
    assert SYNTHETIC_BOUNDARY["apple_gpu_metal_this_host"] is False


@requires_metal
def test_a_metal_run_with_the_host_timer_is_a_measurement():
    report = measure_report(CONFIG, backend=metal_runtime())

    assert (report["measured"], report["label"]) == (True, MEASUREMENT_LABEL)
    assert report["measurement_scope"] == "apple_gpu_metal_this_host"
    assert report["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert (report["backend"], report["timer"]) == (
        "torch.mps.compile_shader",
        "time.perf_counter_ns",
    )
    assert report["environment"]["platform_system"] == "Darwin"
    assert report["summary"]["vector_add_exact_cases"] == 3
    assert report["summary"]["rgb_to_gray_max_abs_difference_levels"] == 0


@requires_metal
def test_a_metal_run_with_an_injected_clock_is_synthetic():
    report = measure_report(CONFIG, backend=metal_runtime(), clock=scripted_clock())

    assert (report["backend"], report["timer"]) == (
        "torch.mps.compile_shader",
        "injected",
    )
    assert (report["measured"], report["label"]) == (False, SYNTHETIC_LABEL)
