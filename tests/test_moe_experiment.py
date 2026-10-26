from dataclasses import replace

import pytest
import torch

from ai_journey.moe_experiment import (
    build_matched_moe_experiment_model,
    run_moe_document_trial,
)
from test_moe_experiment_plan import plan

TRAIN = b"first authored training sentence\nsecond distinct training document\n"
HELDOUT = b"heldout public material\nsecond heldout control\n"


def test_actual_trial_matches_shared_tensors_and_replays_updates_and_routes():
    controls = plan(steps=2)
    rng = torch.get_rng_state().clone()
    result = run_moe_document_trial(controls, TRAIN, HELDOUT, seed=53)
    assert result == run_moe_document_trial(controls, TRAIN, HELDOUT, seed=53)
    assert torch.equal(torch.get_rng_state(), rng)
    assert len({arm["initial_shared_sha256"] for arm in result["arms"]}) == 1
    for arm, config in zip(result["arms"], controls.arms, strict=True):
        assert arm["initial_model_sha256"] != arm["final_model_sha256"]
        assert arm["consumed_target_tokens"] == controls.steps * controls.batch_size * 4
        assert arm["heldout"]["capacity_policy"] == "dropless_eval"
        for row in arm["trace"]:
            assert row["objective"] == pytest.approx(
                row["ce_loss"]
                + config.balance_weight * row["balance_loss"]
                + config.z_weight * row["router_z_loss"],
                rel=2e-6,
            )
    assert (
        result["arms"][1]["routing_drift"][0]["observed"]["assignments"]
        == 2 * result["documents"]["heldout_target_tokens"]
    )
    assert result["arms"][0]["routing_drift"] == []


def test_named_initialization_does_not_shift_shared_attention_or_embeddings():
    controls = plan()
    dense = build_matched_moe_experiment_model(controls.arms[0].config, 53)
    sparse = build_matched_moe_experiment_model(controls.arms[1].config, 53)
    for name, tensor in dense.state_dict().items():
        if ".mlp." not in name:
            assert torch.equal(tensor, sparse.state_dict()[name])
    assert sparse.wte.weight is sparse.lm_head.weight


def test_trial_rejects_undeclared_seed_and_content_leakage():
    with pytest.raises(ValueError, match="declared seed"):
        run_moe_document_trial(plan(), TRAIN, HELDOUT, seed=99)
    with pytest.raises(ValueError, match="content-disjoint"):
        run_moe_document_trial(plan(), TRAIN, TRAIN, seed=53)


def test_whole_job_is_admitted_before_any_model_allocation(monkeypatch):
    import ai_journey.moe_experiment as module

    monkeypatch.setattr(
        module,
        "build_matched_moe_experiment_model",
        lambda *args: pytest.fail("must reject before allocation"),
    )
    controls = plan(seeds=(1, 2, 3, 4), steps=32, batch_size=8)
    with pytest.raises(ValueError, match="work_units"):
        run_moe_document_trial(controls, TRAIN, b"x" * 8190 + b"\n", seed=1)


def test_objective_treatments_share_initial_weights_but_measure_real_interventions():
    controls = plan()
    no_aux = replace(controls.arms[1], name="no-aux", balance_weight=0, z_weight=0)
    controls = replace(controls, arms=(*controls.arms, no_aux))
    result = run_moe_document_trial(controls, TRAIN, HELDOUT, seed=53)
    left, right = result["arms"][1:]
    assert left["initial_model_sha256"] == right["initial_model_sha256"]
    assert left["trace"][0]["ce_loss"] == right["trace"][0]["ce_loss"]
    assert left["trace"][0]["objective"] != right["trace"][0]["objective"]
    assert left["final_model_sha256"] != right["final_model_sha256"]
