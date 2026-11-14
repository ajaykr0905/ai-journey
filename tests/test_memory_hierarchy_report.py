import itertools
import time

import pytest

from ai_journey.memory_access_report import report_digest
from ai_journey.memory_hierarchy_config import config_from_mapping
from ai_journey.memory_hierarchy_report import (
    MEASUREMENT_BOUNDARY,
    REPORT_DIGEST_FIELD,
    SYNTHETIC_BOUNDARY,
    build_report,
    measure_report,
)
from ai_journey.memory_hierarchy_stream import MemoryHierarchyError
from test_memory_hierarchy_config import make_config_mapping

SECTIONS = {
    "schema_version",
    "kind",
    "measured",
    "measurement_scope",
    "evidence_boundary",
    "config",
    "config_sha256",
    "timer",
    "percentile_method",
    "environment",
    "planned_work",
    "stream",
    "access",
    "matmul",
    "roofline",
    "littles_law",
    "run",
    "summary",
}


def make_config(**changes):
    return config_from_mapping({**make_config_mapping(), **changes})


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


def synthetic_report():
    return measure_report(make_config(), clock=scripted_clock())


def test_an_injected_clock_report_is_sealed_and_labeled_synthetic():
    report = synthetic_report()

    assert set(report) == SECTIONS | {REPORT_DIGEST_FIELD}
    assert report[REPORT_DIGEST_FIELD] == report_digest(report)
    assert report["kind"] == "memory_hierarchy_cpu_measurement"
    assert (report["timer"], report["measured"]) == ("injected", False)
    assert report["measurement_scope"] == "synthetic_injected_clock"
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert report["config"] == make_config_mapping()
    assert report["config_sha256"] == make_config().fingerprint()
    assert report["planned_work"] == make_config().planned_work().to_mapping()


def test_sections_cover_the_sweep_patterns_matmul_and_derived_views():
    report = synthetic_report()
    stream, access = report["stream"], report["access"]

    assert [(item["kernel"], item["elements"]) for item in stream["measurements"]] == [
        (kernel, elements)
        for elements in (256, 1024, 4096)
        for kernel in ("copy", "scale", "add", "triad")
    ]
    assert [item["name"] for item in stream["accounting"]] == [
        "copy",
        "scale",
        "add",
        "triad",
    ]
    assert list(stream["curves"]) == ["copy", "scale", "add", "triad"]
    assert stream["scalar"] == 3.0 and stream["element_bytes"] == 8
    assert [item["name"] for item in access["measurements"]] == [
        "copy",
        "sequential",
        "strided",
        "random",
    ]
    assert (access["stride_bytes"], access["source_bytes"]) == (128, 4096 * 8)
    assert report["matmul"]["samples_ns"] == [1000, 1000, 1000]
    assert report["roofline"]["placement_elements"] == 4096
    assert report["littles_law"]["latency_measured"] is False
    assert report["littles_law"]["bandwidth"]["elements"] == 4096


def test_run_totals_and_summary_reconcile_with_the_sections():
    report = synthetic_report()
    measurements = [
        *report["stream"]["measurements"],
        *report["access"]["measurements"],
        report["matmul"],
    ]
    run, summary = report["run"], report["summary"]

    assert run["max_runtime_ns"] == 60 * 10**9
    assert run["timed_ns"] == sum(sum(item["samples_ns"]) for item in measurements)
    assert run["timed_ns"] < run["elapsed_ns"] <= run["max_runtime_ns"]
    assert summary["stream_measurements"] == 12
    assert summary["access_patterns"] == 4
    assert summary["timed_samples"] == 17 * 3
    triad = report["stream"]["curve_summaries"]["triad"]
    assert (
        summary["triad_largest_gb_per_second"]
        == (triad["largest_median_gb_per_second"])
    )
    assert (
        summary["ridge_point_flops_per_byte"]
        == (report["roofline"]["ridge_point_flops_per_byte"])
    )


def test_a_scripted_clock_reproduces_the_report_byte_for_byte():
    first, second = synthetic_report(), synthetic_report()

    assert first == second
    assert first[REPORT_DIGEST_FIELD] == second[REPORT_DIGEST_FIELD]


def test_host_timer_reports_are_cpu_measurements_with_positive_rates():
    report = measure_report(make_config())

    assert (report["timer"], report["measured"]) == ("time.perf_counter_ns", True)
    assert report["measurement_scope"] == "cpu_wall_clock_this_host"
    assert report["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert all(
        item["median_gb_per_second"] > 0 for item in report["stream"]["measurements"]
    )
    assert report["summary"]["compute_ceiling_flops_per_second"] > 0
    assert 0 < report["run"]["timed_ns"] <= report["run"]["elapsed_ns"]


def test_the_run_budget_stops_a_report_that_runs_too_long():
    config = make_config(
        timing={**make_config_mapping()["timing"], "max_runtime_seconds": 1}
    )

    with pytest.raises(MemoryHierarchyError, match="max_runtime_seconds=1"):
        build_report(config, clock=scripted_clock(step=50_000_000))
    with pytest.raises(TypeError, match="MemoryHierarchyConfig"):
        build_report(make_config_mapping(), clock=time.perf_counter_ns)


def test_both_boundaries_deny_gpu_cache_and_latency_claims():
    for boundary in (MEASUREMENT_BOUNDARY, SYNTHETIC_BOUNDARY):
        assert boundary["gpu_executed"] is False
        assert boundary["cuda_kernel_executed"] is False
        assert boundary["cache_sizes_detected"] is False
        assert boundary["latency_measured"] is False
        assert boundary["portable_performance_claim"] is False
    assert MEASUREMENT_BOUNDARY["cpu_measurement_this_host"] is True
    assert SYNTHETIC_BOUNDARY["cpu_measurement_this_host"] is False
    assert "not a measurement of this or any host" in SYNTHETIC_BOUNDARY["statement"]
    assert "No cache size is detected or claimed" in MEASUREMENT_BOUNDARY["statement"]
