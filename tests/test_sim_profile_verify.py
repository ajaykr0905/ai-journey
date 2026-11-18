import copy

import pytest

from ai_journey.kernel_launch_report import seal_report
from ai_journey.sim_profile_config import (
    SimProfileConfig,
    SimProfileError,
    SimProfileVerificationError,
)
from ai_journey.sim_profile_snapshot import snapshot
from ai_journey.sim_profile_verify import (
    nearest_rank,
    timing_statistics,
    verify_snapshot,
)
from test_sim_profile_fixtures import fixed_load, small_config, stepping_clock


@pytest.fixture(scope="module")
def sealed():
    return snapshot(
        small_config(), "baseline", clock=stepping_clock(4), load_average=fixed_load
    )


@pytest.fixture(scope="module")
def measured():
    return snapshot(small_config(), "baseline", load_average=fixed_load)


def reseal(report, mutate):
    changed = copy.deepcopy(report)
    changed.pop("report_sha256")
    mutate(changed)
    return seal_report(changed)


def test_a_fresh_snapshot_verifies_with_and_without_its_config(sealed, measured):
    summary = verify_snapshot(sealed, config=small_config())

    assert summary["stage"] == "baseline"
    assert summary["measured"] is False
    assert summary["total_median_ns"] == 12
    assert summary["report_sha256"] == sealed["report_sha256"]
    assert summary["top_hotspot"] == sealed["profile"]["hotspots"][0]["function"]
    assert verify_snapshot(measured)["scope"] == "cpu_wall_clock_this_host"


def test_nearest_rank_statistics_never_interpolate():
    samples = [9, 1, 5, 3, 7, 2, 8]

    assert [nearest_rank(samples, p) for p in (25, 50, 75)] == [2, 5, 8]
    assert timing_statistics([4]) == {
        "count": 1,
        "min_ns": 4,
        "p25_ns": 4,
        "median_ns": 4,
        "p75_ns": 4,
        "max_ns": 4,
        "iqr_ns": 0,
    }


def test_an_edited_field_breaks_the_seal(sealed):
    changed = copy.deepcopy(sealed)
    changed["summary"]["total_median_ns"] += 1

    with pytest.raises(SimProfileVerificationError, match="does not match the report"):
        verify_snapshot(changed)


def set_path(*path, value):
    def mutate(report):
        target = report
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def edit_first_function(report):
    report["workloads"][0]["profile"]["functions"][0]["calls"] += 1


def drop_sample(report):
    report["workloads"][1]["timing"]["samples_ns"].pop()


def swap_hotspots(report):
    spots = report["profile"]["hotspots"]
    spots[0], spots[1] = spots[1], spots[0]


def fake_median(report):
    report["workloads"][2]["timing"]["timing"]["median_ns"] = 5


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (set_path("kind", value="other"), "kind"),
        (set_path("stage", value="round-9"), "stage"),
        (set_path("stage", value="final"), "stage"),
        (set_path("timer", value="time.time"), "timer"),
        (set_path("measured", value=True), "measured"),
        (set_path("label", value="host-specific CPU measurement"), "label"),
        (set_path("evidence_boundary", "gpu_executed", value=True), "evidence_boundary"),
        (set_path("load_average_after", value=[1.0, 2.0]), "load_average_after"),
        (set_path("load_average_before", value=[1.0, -2.0, 0.0]), "load_average_before"),
        (set_path("semantics", "sha256", value="0" * 64), "semantics.sha256"),
        (set_path("semantics", "guarded_output_order_free", value=False), "order_free"),
        (set_path("config", "timing", "repeats", value=6), "workload_sha256"),
        (set_path("workload_sha256", value="0" * 64), "workload_sha256"),
        (set_path("workloads", 0, "output_sha256", value="0" * 64), "output_sha256"),
        (set_path("workloads", 1, "launch", "grid", value=[1, 1, 1]), "launch"),
        (set_path("workloads", 0, "reads", value=1), "reads"),
        (edit_first_function, "profile"),
        (drop_sample, "one sample per repeat"),
        (fake_median, "timing.timing"),
        (swap_hotspots, "profile.hotspots"),
        (set_path("summary", "top_hotspot", value="ai_journey.x:y"), "summary"),
        (set_path("environment", "python_version", value=3), "environment"),
        (lambda report: report.pop("ranking"), "fields mismatch"),
    ],
)  # fmt: skip
def test_resealed_tampering_is_named(sealed, mutate, message):
    with pytest.raises(SimProfileVerificationError, match=message):
        verify_snapshot(reseal(sealed, mutate))


def test_a_snapshot_binds_only_to_its_own_workloads_and_stages(sealed):
    other = small_config()
    mapping = other.to_mapping()
    mapping["workloads"]["vector_add"]["n"] = 301

    with pytest.raises(SimProfileError, match="other workloads"):
        verify_snapshot(sealed, config=SimProfileConfig.from_mapping(mapping))
    round_one = reseal(sealed, set_path("stage", value="round-1"))
    with pytest.raises(SimProfileError, match="not a stage"):
        verify_snapshot(round_one, config=small_config())
    with pytest.raises(TypeError, match="SimProfileConfig"):
        verify_snapshot(sealed, config="config")
    with pytest.raises(SimProfileVerificationError, match="JSON object"):
        verify_snapshot([sealed])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("out_of_bounds_reads", 0),
        ("out_of_bounds_writes", 309),
        ("write_conflicts", 0),
        ("read_write_conflicts", 139),
        ("runs", 2),
    ],
)
def test_resealed_misuse_totals_must_match_the_pinned_digest(sealed, key, value):
    # The totals follow from the pinned records, so a snapshot that keeps the
    # digest but reports fewer races or out-of-bounds accesses is tampered.
    with pytest.raises(SimProfileVerificationError, match=f"semantics.{key}"):
        verify_snapshot(reseal(sealed, set_path("semantics", key, value=value)))
