import copy

import pytest

from ai_journey.decoder_variant_comparison import (
    decoder_comparison_digest,
    run_decoder_document_comparison,
    summarize_decoder_trials,
)
from test_decoder_variant_document_experiment import TRAIN, HELDOUT
from test_decoder_variant_plan import plan


def test_comparison_pairs_seeds_before_summarizing_and_replays():
    controls = plan(seeds=(7, 13))
    report = run_decoder_document_comparison(controls, TRAIN, HELDOUT)
    assert report == run_decoder_document_comparison(controls, TRAIN, HELDOUT)
    assert report["report_sha256"] == decoder_comparison_digest(report)
    deltas = report["summary"]["paired_deltas"][0]
    expected = [
        trial["arms"][1]["heldout"]["mean_nll"]
        - trial["arms"][0]["heldout"]["mean_nll"]
        for trial in report["trials"]
    ]
    assert [row["delta_nll"] for row in deltas["per_seed"]] == expected
    assert deltas["delta_nll"]["mean"] == sum(expected) / 2
    assert deltas["delta_nll"]["count"] == 2
    changed = copy.deepcopy(report)
    changed["trials"][0]["arms"][0]["final_model_fingerprint"] = "0" * 64
    assert decoder_comparison_digest(changed) != report["report_sha256"]


def test_summary_rejects_incomplete_seeds_or_unequal_target_coverage():
    controls = plan(seeds=(7, 13))
    report = run_decoder_document_comparison(controls, TRAIN, HELDOUT)
    with pytest.raises(ValueError, match="seed"):
        summarize_decoder_trials(controls, report["trials"][:1])
    trials = copy.deepcopy(report["trials"])
    trials[1]["arms"][0]["heldout"]["target_tokens"] += 1
    with pytest.raises(ValueError, match="coverage"):
        summarize_decoder_trials(controls, trials)


def test_single_seed_does_not_fabricate_a_standard_deviation():
    report = run_decoder_document_comparison(plan(), TRAIN, HELDOUT)
    assert report["summary"]["heldout_nll"]["baseline"]["sample_stddev"] is None
