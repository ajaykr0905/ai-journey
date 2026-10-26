from dataclasses import replace

import pytest
import torch

from ai_journey.decoder_variant_document_experiment import run_decoder_document_trial
from test_decoder_variant_plan import plan

TRAIN = b"first public training document\nsecond training document"
HELDOUT = b"first distinct heldout\nsecond heldout"


def test_document_trial_replays_complete_model_training_and_coverage_evidence():
    controls = plan(steps=2)
    rng = torch.get_rng_state().clone()
    first = run_decoder_document_trial(controls, TRAIN, HELDOUT, seed=52)
    second = run_decoder_document_trial(controls, TRAIN, HELDOUT, seed=52)
    assert first == second
    assert torch.equal(torch.get_rng_state(), rng)
    assert first["plan_fingerprint"] == controls.fingerprint()
    assert len(first["batch_schedule_sha256"]) == 64
    for arm in first["arms"]:
        assert arm["initial_model_fingerprint"] != arm["final_model_fingerprint"]
        assert len(arm["final_model_fingerprint"]) == 64
        assert arm["consumed_target_tokens"] == 16
        assert arm["heldout"]["target_tokens"] == sum(
            len(line) for line in HELDOUT.splitlines()
        )
        assert [row["step"] for row in arm["trace"]] == [1, 2]
        assert arm["optimizer_groups"]["trainable_parameters"] == arm["parameter_count"]


def test_document_trial_rejects_undeclared_seed_and_leakage():
    controls = plan()
    with pytest.raises(ValueError, match="seed"):
        run_decoder_document_trial(controls, TRAIN, HELDOUT, seed=7)
    with pytest.raises(ValueError, match="content-disjoint"):
        run_decoder_document_trial(controls, TRAIN, TRAIN + b"\n", seed=52)


def test_plan_rejects_explicit_swiglu_allocation_beyond_budget():
    controls = plan()
    wide = replace(controls.variants[0][1], mlp_kind="swiglu", swiglu_hidden_dim=10**12)
    with pytest.raises(ValueError, match="SwiGLU width"):
        replace(
            controls,
            variants=(controls.variants[0], ("wide", wide)),
            declared_changes=("mlp_kind", "swiglu_hidden_dim"),
        )
