import pytest

from ai_journey.gpu_timing_clocks import CLOCKS, DEVICE_DOMAIN, EVENT, HOST_DOMAIN
from ai_journey.gpu_timing_config import config_from_mapping
from ai_journey.gpu_timing_fit import FIXED_DOMINANT, SIZE_DOMINANT
from ai_journey.gpu_timing_report import (
    MEASUREMENT_BOUNDARY,
    REPORT_KIND,
    build_report,
    measure_report,
)
from ai_journey.matmul_metal_runtime import SIDES
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import report_digest
from test_gpu_timing_config import small
from test_gpu_timing_runtime import timing_runtime
from test_gpu_timing_sweep import StandInBackend
from test_kernel_metal_runtime import requires_metal

LOAD = (2.5, 3.0, 3.25)


def stand_in_report(**changes):
    backend = StandInBackend()
    config = config_from_mapping(small(**changes))
    return measure_report(
        config, backend=backend, clock=backend.clock, load_average=lambda: LOAD
    )


def test_a_stand_in_report_is_sealed_and_labeled_synthetic():
    report = stand_in_report()

    assert report["kind"] == REPORT_KIND
    assert report["measured"] is False
    assert report["measurement_scope"] == "synthetic_backend_or_clock"
    assert report["label"] == "synthetic timing, not a measurement"
    assert report["backend"] == "injected" and report["timer"] == "injected"
    assert report["evidence_boundary"]["synthetic_timing"] is True
    assert report["report_sha256"] == report_digest(report)
    assert canonical_json_bytes(report)


def test_the_report_names_both_clock_domains_and_the_schedule():
    report = stand_in_report()

    clocks = report["clocks"]
    assert clocks["order"] == list(CLOCKS)
    assert set(clocks["domains"].values()) == {DEVICE_DOMAIN, HOST_DOMAIN}
    assert "shifted by r" in clocks["schedule"]
    for point in report["benchmark"]["points"]:
        for side in SIDES:
            entry = point["sides"][side]["clocks"]
            assert entry[EVENT]["domain"] == DEVICE_DOMAIN
            assert "elapsed_ms" in entry[EVENT]


def test_the_report_records_load_averages_before_and_after():
    report = stand_in_report()

    assert report["host_load"]["before"] == list(LOAD)
    assert report["host_load"]["after"] == list(LOAD)
    assert "os.getloadavg()" in report["host_load"]["source"]
    assert report["summary"]["load_average_before"] == list(LOAD)


def test_the_summary_finds_where_work_overtakes_the_fixed_overhead():
    shapes = [[1, 1, 1], [16, 16, 16], [128, 128, 128], [256, 256, 256]]

    report = stand_in_report(sweep__shapes=shapes)

    summary = report["summary"]
    assert summary["sweep_points"] == 4
    assert summary["largest_point"] == {"index": 3, "m": 256, "n": 256, "k": 256}
    for side in SIDES:
        entry = summary["sides"][side]
        assert entry["largest_event_regime"] == SIZE_DOMINANT
        index = entry["first_size_dominant_index"]
        assert index is not None and index > 0
        point = report["analysis"]["points"][index - 1]
        assert point["sides"][side]["clocks"][EVENT]["regime"] == FIXED_DOMINANT
    assert summary["timed_samples"] == report["planned_work"]["timed_samples"]


def test_the_run_totals_count_only_host_samples():
    report = stand_in_report()

    total = 0
    benchmark = report["benchmark"]
    timings = [benchmark["empty"]["timing"]] + [
        point["sides"][side] for point in benchmark["points"] for side in SIDES
    ]
    for timing in timings:
        for clock in ("host_synchronized", "host_dispatch"):
            total += sum(timing["clocks"][clock]["samples_ns"])
    assert report["run"]["timed_host_ns"] == total
    assert report["run"]["elapsed_ns"] > 0


def test_a_report_needs_a_config():
    with pytest.raises(TypeError):
        build_report(small(), backend=StandInBackend())


@requires_metal
def test_a_short_run_on_this_gpu_is_labeled_a_host_specific_measurement():
    config = config_from_mapping(small())

    report = build_report(config, backend=timing_runtime())

    assert report["measured"] is True
    assert report["label"] == "host-specific Apple GPU (Metal) measurement"
    assert report["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert report["timer"] == "time.perf_counter_ns"
    assert len(report["host_load"]["before"]) == 3
