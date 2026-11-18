from pathlib import Path

import pytest

from ai_journey.kernel_metal_report import (
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from ai_journey.matmul_bottleneck_config import load_config
from ai_journey.matmul_bottleneck_mapping import WORSE_AS_MODELED
from ai_journey.matmul_bottleneck_report import (
    DEFINITIONS,
    MEASUREMENT_BOUNDARY,
    SYNTHETIC_BOUNDARY,
    build_report,
    reasons_mapping,
    top_bottleneck,
)
from ai_journey.matmul_bottleneck_rule import (
    DISPATCH_BOUND,
    INDETERMINATE,
    LATENCY_BOUND,
    classify,
)
from ai_journey.matmul_bottleneck_sources import join_points
from ai_journey.matmul_bottleneck_timing import timing_from_mapping
from ai_journey.memory_access_report import DIGEST_FIELD, report_digest
from test_matmul_bottleneck_sources import recorded_sources, synthetic_timing
from test_matmul_bottleneck_timing import synthetic_timing_mapping

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config/day-75-matmul-bottleneck.json")
NAIVE = ("matmul_x_columns", "matmul_x_rows")


def synthetic_report():
    """The recorded Day 74 sources with the synthetic timing input."""

    return build_report(recorded_sources(), synthetic_timing(), CONFIG)


def measured_timing():
    mapping = synthetic_timing_mapping()
    mapping.update(measured=True, label=MEASUREMENT_LABEL)
    return timing_from_mapping(mapping)


def test_a_synthetic_timing_input_makes_a_synthetic_report():
    report = synthetic_report()

    assert (report["measured"], report["label"], report["measurement_scope"]) == (
        False,
        SYNTHETIC_LABEL,
        SYNTHETIC_SCOPE,
    )
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert report[DIGEST_FIELD] == report_digest(report)
    assert report["config"] == CONFIG.to_mapping()
    assert report["config_sha256"] == CONFIG.fingerprint()
    assert report["definitions"] == DEFINITIONS
    assert report["reasons"] == reasons_mapping()
    assert list(report["reasons"])[0] == "dispatch_time_above_host_time"


def test_a_measured_gap_and_timing_make_a_measured_report_that_disclaims_counters():
    sources = recorded_sources()
    report = build_report(sources, measured_timing(), CONFIG)
    boundary = report["evidence_boundary"]

    assert (report["measured"], report["label"], report["measurement_scope"]) == (
        True,
        MEASUREMENT_LABEL,
        MEASUREMENT_SCOPE,
    )
    assert boundary == MEASUREMENT_BOUNDARY
    for flag in ("hardware_counters_used", "nsight_used", "instruments_used"):
        assert boundary[flag] is False
    assert boundary["inference_from_timings"] is True
    assert "No hardware counters, Nsight or Instruments" in boundary["statement"]
    gap_sources = sources.gap_report["sources"]["gpu_ceilings"]
    ceiling = report["sources"]["gpu_ceilings"]
    assert ceiling["load_average_before"] == gap_sources["load_average_before"]
    timing = report["sources"]["timing"]
    assert timing["load_average_before"] == [1.0, 1.5, 2.0]
    assert (timing["entries"], timing["entries_used"]) == (8, 4)
    assert timing["fastest_calls"]["matmul_x_columns"]["median_host_ns"] == 120_000
    assert timing["input_sha256"] == measured_timing().digest()


def test_each_decision_is_the_rule_applied_to_its_joined_inputs():
    sources = recorded_sources()
    report = synthetic_report()
    joined = join_points(sources, synthetic_timing(), CONFIG.rule)

    for item in joined:
        entry = report["points"][item.index]["kernels"][item.kernel]
        assert entry["decision"] == classify(item.inputs, CONFIG.rule).to_mapping()
        assert entry["inputs"] == {
            "median_gpu_ns": item.timing.gpu_ns,
            "median_host_ns": item.timing.host_ns,
            "median_dispatch_ns": item.timing.dispatch_ns,
        }
    first = report["points"][0]
    assert (first["m"], first["n"], first["k"]) == (256, 256, 256)
    assert first["intensity"]["model"] == "ideal_reuse"
    assert first["threadgroups"] == {
        "matmul_x_columns": [8, 32],
        "matmul_x_rows": [8, 32],
    }
    assert report["geometry"] == {
        "group_size": [32, 8],
        "simd_widths": {"matmul_x_columns": 32, "matmul_x_rows": 32},
    }


def test_the_synthetic_times_reach_three_steps_of_the_rule():
    report = synthetic_report()
    square, ragged = (point["kernels"] for point in report["points"])

    for kernel in NAIVE:
        assert square[kernel]["decision"]["reason"] == "below_both_roofs"
    # 691 us of steady work per call cannot fill a 1500 us call.
    assert ragged["matmul_x_columns"]["decision"]["bottleneck"] == DISPATCH_BOUND
    assert ragged["matmul_x_rows"]["decision"]["reason"] == (
        "gpu_time_disagrees_with_steady_time"
    )
    summary = report["summary"]
    assert summary["points"] == 2
    assert summary["mapping_verdicts"][WORSE_AS_MODELED] == 2
    # One point each way is a tie, so neither kernel gets a top name.
    for kernel in NAIVE:
        top = summary["top_bottleneck"][kernel]
        assert (top["top"], top["reason"], top["top_points"]) == (
            INDETERMINATE,
            "tie",
            1,
        )
        assert top["counts"][LATENCY_BOUND] == 1


@pytest.mark.parametrize(
    ("names", "top", "reason"),
    [
        ([LATENCY_BOUND, LATENCY_BOUND, DISPATCH_BOUND], LATENCY_BOUND, "most_points"),
        ([LATENCY_BOUND, DISPATCH_BOUND], INDETERMINATE, "tie"),
        ([INDETERMINATE], INDETERMINATE, "most_points"),
    ],
)
def test_the_top_bottleneck_is_the_most_common_name_and_a_tie_claims_none(
    names, top, reason
):
    result = top_bottleneck(names)

    assert (result["top"], result["reason"]) == (top, reason)
    assert result["top_points"] == max(names.count(name) for name in names)
    assert sum(result["counts"].values()) == len(names)
