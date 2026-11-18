import copy

import pytest

from ai_journey.kernel_launch_report import seal_report
from ai_journey.sim_profile_config import (
    SimProfileConfig,
    SimProfileError,
    SimProfileVerificationError,
)
from ai_journey.sim_profile_rounds import rounds_report
from ai_journey.sim_profile_rounds_verify import verify_rounds
from ai_journey.sim_profile_snapshot import snapshot
from test_sim_profile_fixtures import fixed_load, small_config, stepping_clock


def take(config, stage, step):
    return snapshot(config, stage, clock=stepping_clock(step), load_average=fixed_load)


@pytest.fixture(scope="module")
def sealed():
    baseline = take(small_config(), "baseline", 9)
    target = baseline["profile"]["hotspots"][0]["function"]
    config = small_config([{"target": target, "change": "One change."}])
    return config, rounds_report(config, [baseline, take(config, "round-1", 6)])


def reseal(report, mutate):
    changed = copy.deepcopy(report)
    changed.pop("report_sha256")
    mutate(changed)
    return seal_report(changed)


def test_a_fresh_rounds_report_verifies(sealed):
    config, report = sealed
    summary = verify_rounds(report, config=config)

    assert summary == {
        "kind": "sim_profile_rounds",
        "report_sha256": report["report_sha256"],
        "measured": False,
        "scope": "synthetic_injected_clock",
        "rounds": 1,
        "round_speedups": [1.5],
        "speedup": 1.5,
    }


def bump_speedup(report):
    report["rounds"][0]["total"]["speedup"] = 2.0


def bump_workload(report):
    report["rounds"][0]["workloads"][1]["after_median_ns"] -= 1


def rename_top(report):
    report["rounds"][0]["top_hotspot_before"] = "ai_journey.x:y"


def fake_overall(report):
    report["overall"]["speedup"] = 3.0


def tamper_snapshot(report):
    report["snapshots"][1]["summary"]["total_median_ns"] = 1


def swap_snapshots(report):
    report["snapshots"].reverse()


def claim_measured(report):
    report["measured"] = True


def drop_round(report):
    report["config"]["rounds"] = []


def move_load(report):
    report["stages"][0]["load_average_before"] = [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (bump_speedup, r"rounds\[0\]"),
        (bump_workload, r"rounds\[0\]"),
        (rename_top, r"rounds\[0\]"),
        (fake_overall, "overall"),
        (tamper_snapshot, r"snapshots\[1\]"),
        (swap_snapshots, r"snapshots\[0\]"),
        (claim_measured, "measured"),
        (drop_round, "snapshots must hold one report per stage|names no round"),
        (move_load, "stages"),
        (lambda report: report.pop("overall"), "fields mismatch"),
    ],
)
def test_resealed_tampering_is_named(sealed, mutate, message):
    _, report = sealed

    with pytest.raises(SimProfileVerificationError, match=message):
        verify_rounds(reseal(report, mutate))


def test_an_unsealed_edit_and_another_config_are_refused(sealed):
    config, report = sealed
    edited = copy.deepcopy(report)
    edited["overall"]["speedup"] = 9.0
    mapping = config.to_mapping()
    mapping["rounds"][0]["change"] = "Another change."

    with pytest.raises(SimProfileVerificationError, match="does not match the report"):
        verify_rounds(edited)
    with pytest.raises(SimProfileError, match="another config"):
        verify_rounds(report, config=SimProfileConfig.from_mapping(mapping))
    with pytest.raises(TypeError, match="SimProfileConfig"):
        verify_rounds(report, config="config")
