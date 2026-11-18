import pytest

from ai_journey import sim_profile_snapshot
from ai_journey.kernel_launch_report import report_digest
from ai_journey.memory_access_benchmark import INJECTED_TIMER, TIMER
from ai_journey.sim_profile_config import SimProfileError
from ai_journey.sim_profile_harness import FunctionProfile, merge_functions
from ai_journey.sim_profile_semantics import SEMANTICS_SHA256
from ai_journey.sim_profile_snapshot import (
    GETITEM,
    MEASUREMENT_BOUNDARY,
    SYNTHETIC_BOUNDARY,
    build_snapshot,
    snapshot,
)
from test_sim_profile_fixtures import LOAD, fixed_load, small_config, stepping_clock


@pytest.fixture(scope="module")
def synthetic():
    return snapshot(
        small_config(), "baseline", clock=stepping_clock(5), load_average=fixed_load
    )


def calls(entry):
    return [(item["function"], item["calls"]) for item in entry["profile"]["functions"]]


def test_a_synthetic_snapshot_is_labeled_and_sealed(synthetic):
    assert synthetic["report_sha256"] == report_digest(synthetic)
    assert synthetic["stage"] == "baseline"
    assert synthetic["timer"] == INJECTED_TIMER
    assert synthetic["measured"] is False
    assert synthetic["scope"] == "synthetic_injected_clock"
    assert synthetic["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert synthetic["load_average_before"] == list(LOAD)
    assert synthetic["semantics"]["sha256"] == SEMANTICS_SHA256
    assert synthetic["summary"]["total_median_ns"] == 15
    assert [w["name"] for w in synthetic["workloads"]] == [
        "vector_add",
        "rgb_to_grayscale",
        "naive_matmul",
    ]
    for entry in synthetic["workloads"]:
        assert entry["output_sha256"] == entry["reference_sha256"]
        assert entry["timing"]["samples_ns"] == [5] * 5


def test_the_merged_profile_and_hotspots_follow_from_the_workloads(synthetic):
    records = [
        FunctionProfile(**item)
        for entry in synthetic["workloads"]
        for item in entry["profile"]["functions"]
    ]
    merged = [record.to_mapping() for record in merge_functions(records)]
    profile = synthetic["profile"]

    assert profile["functions"] == merged
    assert profile["calls"] == sum(item["calls"] for item in merged)
    hotspots = profile["hotspots"]
    assert [item["rank"] for item in hotspots] == list(range(1, 11))
    totals = [item["total_ns"] for item in hotspots]
    assert totals == sorted(totals, reverse=True)
    assert synthetic["summary"]["top_hotspot"] == hotspots[0]["function"]
    vector = dict(calls(synthetic["workloads"][0]))
    assert vector[GETITEM] == synthetic["workloads"][0]["reads"] == 600


def test_call_counts_repeat_across_snapshots(synthetic):
    again = build_snapshot(
        small_config(), "baseline", clock=stepping_clock(3), load_average=fixed_load
    )

    for first, second in zip(synthetic["workloads"], again["workloads"]):
        assert calls(first) == calls(second)


def test_the_host_timer_makes_a_measurement():
    report = build_snapshot(small_config(), "baseline", load_average=fixed_load)

    assert report["timer"] == TIMER
    assert report["measured"] is True
    assert report["label"] == "host-specific CPU measurement"
    assert report["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert MEASUREMENT_BOUNDARY["gpu_executed"] is False
    assert MEASUREMENT_BOUNDARY["hardware_counters_used"] is False


def test_stages_must_belong_to_the_config():
    rounds = [{"target": "ai_journey.kernel_launch_buffer:to_float32", "change": "x"}]

    with pytest.raises(SimProfileError, match="config stages"):
        build_snapshot(small_config(), "round-1", load_average=fixed_load)
    with pytest.raises(TypeError, match="SimProfileConfig"):
        build_snapshot("config", "baseline")
    assert small_config(rounds).stages == ("baseline", "round-1")


def test_a_tree_with_other_misuse_records_gets_no_snapshot(monkeypatch):
    def refuse():
        raise SimProfileError("an optimization changed a semantic")

    monkeypatch.setattr(sim_profile_snapshot, "checked_semantics", refuse)
    with pytest.raises(SimProfileError, match="changed a semantic"):
        build_snapshot(small_config(), "baseline", load_average=fixed_load)
