import itertools
import time

import pytest

from ai_journey.gpu_ceiling_config import config_from_mapping
from ai_journey.gpu_ceiling_report import (
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_LABEL,
    REPORT_DIGEST_FIELD,
    SYNTHETIC_BOUNDARY,
    SYNTHETIC_LABEL,
    build_report,
    measure_report,
    read_load_average,
)
from ai_journey.gpu_ceiling_source import SHADER_SOURCE, GpuCeilingError, source_sha256
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.kernel_metal_report import ENVIRONMENT_FIELDS
from ai_journey.memory_access_report import report_digest
from test_gpu_ceiling_backend import NumpyBackend
from test_gpu_ceiling_config import changed, make_config_mapping
from test_kernel_metal_runtime import requires_metal

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
    "host_load",
    "shaders",
    "planned_work",
    "correctness",
    "benchmark",
    "ceilings",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}


def varied_clock():
    """Advance by a repeating mix of steps so samples and statistics vary."""

    steps = itertools.cycle((997, 1201, 1499, 1999, 2503, 1103, 1789))
    return itertools.accumulate(steps, initial=0).__next__


def loads():
    readings = iter([(3.5, 4.25, 5.0), (3.75, 4.5, 5.0)])
    return lambda: next(readings)


def stand_in_report(**changes):
    settings = {
        "backend": NumpyBackend(),
        "clock": varied_clock(),
        "load_average": loads(),
    }
    return measure_report(CONFIG, **{**settings, **changes})


def test_a_stand_in_report_is_sealed_and_labeled_synthetic():
    report = stand_in_report()

    assert set(report) == TOP_LEVEL
    assert report[REPORT_DIGEST_FIELD] == report_digest(report)
    assert report["kind"] == "gpu_ceiling_measurement"
    assert (report["measured"], report["label"]) == (False, SYNTHETIC_LABEL)
    assert (report["backend"], report["timer"]) == ("injected", "injected")
    assert report["measurement_scope"] == "synthetic_backend_or_clock"
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert report["evidence_boundary"]["vendor_peak_used"] is False
    assert report["config_sha256"] == CONFIG.fingerprint()
    assert set(report["environment"]) == set(ENVIRONMENT_FIELDS)
    assert report["summary"]["label"] == SYNTHETIC_LABEL


def test_the_host_timer_with_a_stand_in_backend_is_still_synthetic():
    report = stand_in_report(clock=time.perf_counter_ns)

    assert report["timer"] == "time.perf_counter_ns"
    assert (report["measured"], report["backend"]) == (False, "injected")


def test_the_report_records_load_averages_before_and_after_the_run():
    report = stand_in_report()

    assert report["host_load"]["before"] == [3.5, 4.25, 5.0]
    assert report["host_load"]["after"] == [3.75, 4.5, 5.0]
    assert "uptime" in report["host_load"]["source"]
    summary = report["summary"]
    assert summary["load_average_before"] == [3.5, 4.25, 5.0]
    assert summary["load_average_after"] == [3.75, 4.5, 5.0]
    for bad in ((1.0, 2.0), (1.0, float("nan"), 2.0), (1, 2, 3), [1.0, 2.0, 3.0]):
        with pytest.raises(GpuCeilingError, match="three nonnegative floats"):
            read_load_average(lambda bad=bad: bad)


def test_cases_points_and_ceilings_cover_the_config_and_reconcile():
    report = stand_in_report()
    work = report["planned_work"]
    correctness, benchmark = report["correctness"], report["benchmark"]

    assert (
        len(correctness["fma"]) + len(correctness["stream"]) + len(correctness["bank"])
        == work["correctness_cases"]
    )
    assert len(benchmark["fma"]) == work["fma_points"] == 6
    assert len(benchmark["stream"]) == work["stream_points"] == 4
    assert len(benchmark["bank"]) == work["bank_points"] == 6
    ceilings = report["ceilings"]
    rates = [point["median_gflops_per_second"] for point in benchmark["fma"]]
    assert ceilings["fma"]["gflops_per_second"] == max(rates)
    assert ceilings["bandwidth"]["buffer_bytes"] == 65536
    assert ceilings["ridge_point_flops_per_byte"] == (
        ceilings["fma"]["gflops_per_second"] / ceilings["bandwidth"]["gb_per_second"]
    )
    assert [row["chains"] for row in ceilings["fma_scaling"]] == [1, 4]
    assert ceilings["bank"]["rows"][0]["stride"] == 1
    assert "not a vendor peak" in ceilings["definition"]
    summary = report["summary"]
    assert summary["timed_samples"] == work["timed_samples"] == 48
    assert summary["bank_finding"] == ceilings["bank"]["finding"]
    run = report["run"]
    timed = sum(
        sum(point["samples_ns"])
        for family in ("fma", "stream", "bank")
        for point in benchmark[family]
    )
    assert run["timed_ns"] == timed <= run["elapsed_ns"] <= run["max_runtime_ns"]


def test_the_shader_section_holds_the_exact_source_and_every_pipeline():
    shaders = stand_in_report()["shaders"]

    assert shaders["source"] == SHADER_SOURCE
    assert shaders["source_sha256"] == source_sha256()
    assert [kernel["name"] for kernel in shaders["kernels"]][-2:] == [
        "bank_rows_32",
        "bank_rows_33",
    ]
    assert shaders["kernels"][-1]["limits"]["static_threadgroup_memory_bytes"] == 14784


def test_a_run_that_overruns_its_budget_stops_without_a_report():
    config = config_from_mapping(changed(("timing", "max_runtime_seconds"), 1))
    clock = itertools.count(0, 10**8).__next__

    with pytest.raises(KernelMetalError, match="max_runtime_seconds=1"):
        build_report(config, backend=NumpyBackend(), clock=clock, load_average=loads())
    with pytest.raises(TypeError, match="GpuCeilingConfig"):
        build_report(make_config_mapping(), backend=NumpyBackend())


@requires_metal
def test_a_run_on_this_gpu_is_a_labeled_measurement():
    report = measure_report(CONFIG)

    assert (report["measured"], report["label"]) == (True, MEASUREMENT_LABEL)
    assert report["backend"] == "torch.mps.compile_shader"
    assert report["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert report["environment"]["platform_system"] == "Darwin"
    assert all(value >= 0 for value in report["host_load"]["before"])
