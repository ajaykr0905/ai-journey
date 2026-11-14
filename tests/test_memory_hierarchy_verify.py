import itertools
import json
from copy import deepcopy

import pytest

from ai_journey.memory_access_report import seal_report
from ai_journey.memory_hierarchy_config import config_from_mapping
from ai_journey.memory_hierarchy_report import REPORT_DIGEST_FIELD, measure_report
from ai_journey.memory_hierarchy_verify import (
    MemoryHierarchyVerificationError,
    verify_report,
)
from test_memory_hierarchy_config import make_config_mapping

CONFIG = config_from_mapping(make_config_mapping())


def varied_clock():
    """Advance by a repeating mix of steps so samples, curves and peaks vary."""

    steps = itertools.cycle((997, 1201, 1499, 1999, 2503, 1103, 1789))
    return itertools.accumulate(steps, initial=0).__next__


SYNTHETIC = measure_report(CONFIG, clock=varied_clock())
MEASURED = measure_report(CONFIG)


def fresh(report=SYNTHETIC):
    return json.loads(json.dumps(report))


def reseal(report):
    unsigned = {
        key: value for key, value in report.items() if key != REPORT_DIGEST_FIELD
    }
    return seal_report(unsigned)


def at(report, path):
    target = report
    for key in path[:-1]:
        target = target[key]
    return target, path[-1]


def tampered(path, change):
    report = fresh()
    target, key = at(report, path)
    target[key] = change(target[key])
    return reseal(report)


def test_reports_verify_and_bind_to_the_config_they_were_measured_from():
    for report, measured, scope in (
        (SYNTHETIC, False, "synthetic_injected_clock"),
        (MEASURED, True, "cpu_wall_clock_this_host"),
    ):
        summary = verify_report(fresh(report), config=CONFIG)
        assert summary == {
            "kind": "memory_hierarchy_cpu_measurement",
            "report_sha256": report[REPORT_DIGEST_FIELD],
            "config_sha256": CONFIG.fingerprint(),
            "measured": measured,
            "scope": scope,
            "stream_measurements": 12,
            "access_patterns": 4,
            "timed_samples": 51,
            "ridge_point_flops_per_byte": report["summary"][
                "ridge_point_flops_per_byte"
            ],
        }
    other = make_config_mapping()
    other["seed"] = 72
    with pytest.raises(MemoryHierarchyVerificationError, match="supplied config"):
        verify_report(fresh(), config=config_from_mapping(other))


@pytest.mark.parametrize(
    ("path", "change", "message"),
    [
        (
            ("stream", "measurements", 0, "median_gb_per_second"),
            lambda v: v * 2,
            "stream copy at 256",
        ),
        (
            ("stream", "measurements", 5, "samples_ns", 0),
            lambda v: v + 1,
            "stream scale at 1024",
        ),
        (
            ("stream", "measurements", 3, "calls_per_sample"),
            lambda v: v + 1,
            "stream triad at 256",
        ),
        (
            ("stream", "measurements", 2, "samples_ns", 1),
            lambda v: 0,
            "positive integers",
        ),
        (
            ("stream", "measurements", 2, "samples_ns", 1),
            lambda v: True,
            "positive integers",
        ),
        (
            ("stream", "measurements", 2, "elements"),
            lambda v: float(v),
            "stream add at 256",
        ),
        (
            ("stream", "measurements", 0, "working_set_bytes"),
            lambda v: v // 2,
            "stream copy",
        ),
        (
            ("stream", "curves", "triad", 1, "median_gb_per_second"),
            lambda v: v + 1,
            "curves",
        ),
        (
            ("stream", "curve_summaries", "add", "largest_to_peak_ratio"),
            lambda v: v / 2,
            "curves",
        ),
        (("stream", "accounting", 3, "reads_per_element"), lambda v: 2, "accounting"),
        (("stream", "scalar"), lambda v: 2.0, "accounting"),
        (("access", "measurements", 2, "order"), lambda v: "rows", "access strided"),
        (
            ("access", "measurements", 3, "median_ns_per_element"),
            lambda v: v / 2,
            "access random",
        ),
        (
            ("access", "measurements", 1, "index_bytes"),
            lambda v: 4,
            "access sequential",
        ),
        (
            ("access", "contrast", "random_to_sequential_time_ratio"),
            lambda v: v * 3,
            "contrast",
        ),
        (("access", "stride_bytes"), lambda v: v * 2, "contrast"),
        (("matmul", "flops_per_call"), lambda v: v - 1, "matmul accounting"),
        (("matmul", "correctness", "tolerance"), lambda v: v * 10, "matmul accounting"),
        (
            ("matmul", "correctness", "max_bound_ratio"),
            lambda v: 1.5,
            "max_bound_ratio",
        ),
        (("roofline", "ridge_point_flops_per_byte"), lambda v: v * 2, "roofline"),
        (("roofline", "placements", 0, "bottleneck"), lambda v: "compute", "roofline"),
        (
            ("roofline", "placements", 3, "fraction_of_attainable"),
            lambda v: 0.5,
            "roofline",
        ),
        (
            ("roofline", "bandwidth_ceilings", 1, "elements"),
            lambda v: v // 4,
            "roofline",
        ),
        (
            ("littles_law", "bandwidth", "derived_bytes_in_flight"),
            lambda v: v * 10,
            "Little's-law",
        ),
        (("littles_law", "latency_measured"), lambda v: True, "Little's-law"),
        (
            ("littles_law", "access_patterns", 3, "derived_elements_in_flight"),
            lambda v: 1.0,
            "Little's-law",
        ),
        (("planned_work", "accounted_bytes"), lambda v: v + 8, "planned work"),
        (("run", "elapsed_ns"), lambda v: 1, "run time"),
        (("run", "elapsed_ns"), lambda v: 61 * 10**9, "run time"),
        (("summary", "timed_samples"), lambda v: v + 1, "summary"),
        (("environment", "platform_machine"), lambda v: "/home/someone", "environment"),
        (("environment", "cpu_count"), lambda v: 0, "cpu_count"),
        (("config_sha256",), lambda v: "0" * 64, "config digest"),
        (("config", "seed"), lambda v: v + 1, "config digest"),
        (("config", "timing", "repeats"), lambda v: 2, "repeats"),
        (("percentile_method",), lambda v: "linear interpolation", "method fields"),
        (("kind",), lambda v: "memory_access_cpu_measurement", "kind"),
    ],
)
def test_resealed_tampering_is_caught_by_recomputation(path, change, message):
    with pytest.raises(MemoryHierarchyVerificationError, match=message):
        verify_report(tampered(path, change))


def test_a_consistent_config_change_still_fails_on_the_measurements():
    report = fresh()
    report["config"]["timing"]["target_bytes_per_sample"] *= 2
    report["config_sha256"] = config_from_mapping(report["config"]).fingerprint()

    with pytest.raises(MemoryHierarchyVerificationError, match="stream copy at 256"):
        verify_report(reseal(report))


def test_unsealed_edits_fail_on_the_digest():
    report = fresh()
    report["summary"]["triad_peak_gb_per_second"] += 1.0

    with pytest.raises(MemoryHierarchyVerificationError, match="digest"):
        verify_report(report)


def test_timer_measured_flag_scope_and_boundary_must_agree():
    promoted = fresh()
    promoted["timer"] = "time.perf_counter_ns"
    with pytest.raises(MemoryHierarchyVerificationError, match="must be measured"):
        verify_report(reseal(promoted))
    promoted["measured"] = True
    with pytest.raises(MemoryHierarchyVerificationError, match="boundary was weakened"):
        verify_report(reseal(promoted))
    claiming = fresh()
    claiming["measured"] = True
    with pytest.raises(MemoryHierarchyVerificationError, match="must not claim"):
        verify_report(reseal(claiming))
    weakened = fresh(MEASURED)
    weakened["evidence_boundary"]["gpu_executed"] = True
    with pytest.raises(MemoryHierarchyVerificationError, match="boundary was weakened"):
        verify_report(reseal(weakened))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report.pop("summary"),
        lambda report: report.update(extra=1),
        lambda report: report["stream"].update(measurements="none"),
        lambda report: report["stream"]["measurements"].pop(),
        lambda report: report["access"]["measurements"].append({}),
        lambda report: report["matmul"].update(correctness=[]),
        lambda report: report["roofline"].clear(),
        lambda report: report.update(run=[1, 2, 3]),
        lambda report: report.update(config={"name": "x"}),
    ],
)
def test_malformed_sections_fail_verification_instead_of_crashing(mutate):
    report = fresh()
    mutate(report)

    with pytest.raises(MemoryHierarchyVerificationError):
        verify_report(reseal(report))


def test_inputs_must_be_reports_and_configs():
    with pytest.raises(MemoryHierarchyVerificationError, match="JSON object"):
        verify_report([SYNTHETIC])
    with pytest.raises(TypeError, match="MemoryHierarchyConfig"):
        verify_report(fresh(), config=make_config_mapping())
    assert verify_report(deepcopy(SYNTHETIC))["measured"] is False
