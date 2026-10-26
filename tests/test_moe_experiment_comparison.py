import copy
from dataclasses import replace

import pytest

from ai_journey.moe_experiment_comparison import (
    run_moe_document_comparison,
    summarize_moe_trials,
)
from test_moe_experiment import TRAIN, HELDOUT
from test_moe_experiment_plan import plan


def test_paired_seed_summary_uses_actual_counts_and_descriptive_deltas():
    controls = plan(seeds=(53, 54))
    report = run_moe_document_comparison(controls, TRAIN, HELDOUT)
    assert report == run_moe_document_comparison(controls, TRAIN, HELDOUT)
    delta = report["summary"]["paired_deltas"][0]
    for row, trial in zip(delta["per_seed"], report["trials"], strict=True):
        assert (
            row["delta_nll"]
            == trial["arms"][1]["heldout"]["mean_nll"]
            - trial["arms"][0]["heldout"]["mean_nll"]
        )
    assert report["summary"]["heldout_nll"]["top2"]["sample_stddev"] is not None
    for row in report["summary"]["training_route_pressure"]["top2"]:
        assert row["attempted_routes"] == 2 * controls.steps * controls.batch_size * 4
        assert row["accepted_routes"] + row["dropped_routes"] == row["attempted_routes"]
    assert "not population confidence" in report["summary"]["uncertainty_policy"]


def test_single_seed_spread_is_null_and_dense_has_no_fabricated_route_pressure():
    report = run_moe_document_comparison(plan(), TRAIN, HELDOUT)
    assert report["summary"]["heldout_nll"]["baseline"]["sample_stddev"] is None
    assert "baseline" not in report["summary"]["training_route_pressure"]


@pytest.mark.parametrize(
    "mutation", ["missing", "reordered", "coverage", "plan", "nan"]
)
def test_summary_rejects_mixed_or_incomplete_experiment_evidence(mutation):
    controls = plan(seeds=(53, 54))
    report = run_moe_document_comparison(controls, TRAIN, HELDOUT)
    trials = copy.deepcopy(report["trials"])
    if mutation == "missing":
        trials.pop()
    elif mutation == "reordered":
        trials.reverse()
    elif mutation == "coverage":
        trials[1]["documents"]["heldout_target_tokens"] += 1
    elif mutation == "plan":
        trials[0]["plan_sha256"] = replace(controls, learning_rate=0.002).fingerprint()
    else:
        trials[0]["arms"][0]["heldout"]["mean_nll"] = float("nan")
    with pytest.raises(ValueError):
        summarize_moe_trials(controls, trials)
