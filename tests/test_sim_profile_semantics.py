import copy

import pytest

from ai_journey import sim_profile_semantics
from ai_journey.sim_profile_config import SimProfileError
from ai_journey.sim_profile_semantics import (
    CASES,
    SEMANTICS_SHA256,
    SEMANTICS_TOTALS,
    checked_semantics,
    semantic_runs,
    semantics_summary,
)


@pytest.fixture(scope="module")
def runs():
    return semantic_runs()


def test_the_misuse_records_match_the_digest_pinned_before_round_one(runs):
    summary = semantics_summary(runs)

    assert summary == {
        "sha256": SEMANTICS_SHA256,
        "runs": 8,
        "out_of_bounds_reads": 826,
        "out_of_bounds_writes": 308,
        "write_conflicts": 464,
        "read_write_conflicts": 140,
        "canonical_records_order_free": True,
        "guarded_output_order_free": True,
    }
    assert {key: summary[key] for key in SEMANTICS_TOTALS} == SEMANTICS_TOTALS
    assert [run["case"] for run in runs] == [case for case in CASES for _ in "ab"]
    assert [run["order"]["kind"] for run in runs] == ["canonical", "permuted"] * 4


def test_the_stencil_records_negative_indices_and_order_dependent_races(runs):
    canonical, permuted = runs[4:6]
    records = canonical["log"]["out_of_bounds"]

    assert records[0] == {
        "buffer": "x",
        "access": "read",
        "index": -1,
        "block": [0, 0, 0],
        "thread": [0, 0, 0],
    }
    assert canonical["log"]["counts"]["read_write_conflicts"] == 70
    assert canonical["output_sha256"] != permuted["output_sha256"]
    assert canonical["log"]["out_of_bounds"] != permuted["log"]["out_of_bounds"]
    assert (
        canonical["log"]["canonical_out_of_bounds"]
        == permuted["log"]["canonical_out_of_bounds"]
    )


def test_a_lost_or_reordered_record_changes_the_digest(runs):
    lost = copy.deepcopy(runs)
    lost[0]["log"]["out_of_bounds"].pop()
    swapped = copy.deepcopy(runs)
    records = swapped[2]["log"]["write_conflicts"]
    records[0], records[1] = records[1], records[0]

    for changed in (lost, swapped):
        assert semantics_summary(changed)["sha256"] != SEMANTICS_SHA256


def test_a_simulator_with_other_records_is_refused(monkeypatch, runs):
    assert checked_semantics()["sha256"] == SEMANTICS_SHA256

    changed = copy.deepcopy(runs)
    changed[5]["log"]["read_write_conflicts"].pop()
    monkeypatch.setattr(sim_profile_semantics, "semantic_runs", lambda: changed)
    with pytest.raises(SimProfileError, match="changed a semantic"):
        checked_semantics()
