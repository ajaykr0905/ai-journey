import copy

import pytest

from ai_journey.sft_finetune_contract import SFTFineTuneConfig, SFTFineTuneError
from ai_journey.sft_finetune_experiment import LIMITATIONS
from ai_journey.sft_finetune_report import (
    build_finetune_report,
    validate_finetune_report,
)


def result():
    score = {"nll_sum": 4.0, "target_tokens": 2, "mean_nll": 2.0, "perplexity": 7.0}
    digest = "a" * 64
    return {
        "config": SFTFineTuneConfig(steps=1, checkpoint_interval=1).as_mapping(),
        "dataset": {
            "sha256": digest,
            "examples": 2,
            "train_examples": 1,
            "validation_examples": 1,
        },
        "schedule_sha256": digest,
        "initial_model": {"model_sha256": digest},
        "final_model": {"model_sha256": digest},
        "training": {"optimizer_steps": 1, "final_model_sha256": digest},
        "comparison": {
            "before": {"score": score},
            "after": {"score": score},
            "delta_mean_nll": 0.0,
        },
        "runtime": {"device": "cpu", "threads": 1},
        "limitations": LIMITATIONS,
    }


def test_report_seals_and_validates_consistent_experiment_evidence():
    report = build_finetune_report(result())
    assert validate_finetune_report(report) == report


def test_report_rejects_tampered_content():
    report = build_finetune_report(result())
    damaged = copy.deepcopy(report)
    damaged["result"]["training"]["optimizer_steps"] = 0
    with pytest.raises(SFTFineTuneError, match="digest"):
        validate_finetune_report(damaged)
