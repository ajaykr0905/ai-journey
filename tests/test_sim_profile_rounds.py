import copy

import pytest

from ai_journey.kernel_launch_report import report_digest, seal_report
from ai_journey.sim_profile_config import SimProfileConfig, SimProfileError
from ai_journey.sim_profile_rounds import build_rounds, rounds_report, speedup
from ai_journey.sim_profile_snapshot import snapshot
from test_sim_profile_fixtures import (
    fixed_load,
    round_targets,
    small_config,
    stepping_clock,
)


def take(config, stage, step):
    return snapshot(config, stage, clock=stepping_clock(step), load_average=fixed_load)


@pytest.fixture(scope="module")
def loop():
    baseline = take(small_config(), "baseline", 12)
    first, second = round_targets(baseline, 2)
    config = small_config(
        [
            {"target": first, "change": "First change."},
            {"target": second, "change": "Second change."},
        ]
    )
    return config, [baseline, take(config, "round-1", 8), take(config, "round-2", 6)]


def test_speedups_divide_medians_and_round_to_three_decimals():
    assert speedup(12, 8) == 1.5
    assert speedup(2, 3) == 0.667
    with pytest.raises(SimProfileError, match="positive"):
        speedup(0, 3)
    with pytest.raises(TypeError, match="integers"):
        speedup(1.0, 3)


def test_rounds_compare_each_stage_with_the_one_before(loop):
    config, snapshots = loop
    report = rounds_report(config, snapshots)

    assert report["report_sha256"] == report_digest(report)
    assert report["kind"] == "sim_profile_rounds"
    assert report["measured"] is False
    assert [stage["stage"] for stage in report["stages"]] == list(config.stages)
    first, second = report["rounds"]
    baseline = snapshots[0]
    ranks = {spot["function"]: spot["rank"] for spot in baseline["profile"]["hotspots"]}
    assert (first["before_stage"], first["after_stage"]) == ("baseline", "round-1")
    assert first["top_hotspot_before"] == baseline["summary"]["top_hotspot"]
    assert first["target_before"]["rank"] == ranks[first["target"]]
    assert second["target_before"]["rank"] is not None
    assert first["total"] == {
        "before_median_ns": 36,
        "after_median_ns": 24,
        "speedup": 1.5,
    }
    assert [w["speedup"] for w in second["workloads"]] == [1.333] * 3
    assert report["overall"]["speedup"] == 2.0
    assert report["snapshots"] == snapshots
    calls = first["target_before"]["calls"]
    assert calls == first["target_after"]["calls"] > 0


def test_round_targets_skip_ranked_entries_that_no_round_may_target():
    # The harness sums every generated dataclass __init__ into one entry, so
    # it can rank first in a small profile. No round may target it.
    names = [
        "<string>:__init__",
        "builtins:<method 'add' of 'set' objects>",
        "ai_journey.kernel_launch_config:Index3.__post_init__",
        "ai_journey.kernel_launch_simulator:launch",
        "ai_journey.kernel_launch_buffer:DeviceBuffer.__getitem__",
    ]
    report = {"profile": {"hotspots": [{"function": name} for name in names]}}

    assert round_targets(report, 2) == names[2:4]
    with pytest.raises(SimProfileError, match="module:qualname"):
        small_config([{"target": names[0], "change": "One change."}])


def test_a_target_that_the_profile_did_not_rank_is_refused(loop):
    config, snapshots = loop
    mapping = config.to_mapping()
    mapping["rounds"][0]["target"] = "ai_journey.kernel_launch_config:ceil_div"

    other = SimProfileConfig.from_mapping(mapping)
    with pytest.raises(SimProfileError, match="not a hotspot of stage baseline"):
        build_rounds(other, snapshots)


def test_snapshots_must_cover_the_stages_in_order_with_one_clock(loop):
    config, snapshots = loop

    with pytest.raises(SimProfileError, match="give one snapshot each"):
        build_rounds(config, snapshots[:2])
    with pytest.raises(SimProfileError, match="follow the stages"):
        build_rounds(config, [snapshots[1], snapshots[0], snapshots[2]])
    measured = snapshot(config, "round-2", load_average=fixed_load)
    with pytest.raises(SimProfileError, match="same clock"):
        build_rounds(config, [*snapshots[:2], measured])
    moved = copy.deepcopy(snapshots[2])
    moved.pop("report_sha256")
    moved["environment"]["platform_machine"] = "other"
    with pytest.raises(SimProfileError, match="same environment"):
        build_rounds(config, [*snapshots[:2], seal_report(moved)])
    with pytest.raises(SimProfileError, match="names no round"):
        build_rounds(small_config(), snapshots[:1])
