from dataclasses import replace

import pytest
import torch

from ai_journey.gqa_cli import default_gqa_plan
from ai_journey.gqa_experiment import build_controlled_gqa_model, run_gqa_comparison
from ai_journey.gqa_experiment_report import (
    verify_gqa_experiment_report,
    replay_gqa_experiment_report,
)
from ai_journey.moe_experiment import build_matched_moe_experiment_model
from test_moe_experiment import TRAIN, HELDOUT


@pytest.mark.parametrize("heads", [4, 2, 1])
def test_initializer_copies_exact_q_common_and_explicit_contiguous_kv_group_means(
    heads,
):
    controls = default_gqa_plan()
    config = next(config for config in controls.arms if config.kv_heads == heads)
    reference = build_matched_moe_experiment_model(controls.decoder, 53).eval()
    rng = torch.get_rng_state().clone()
    model, evidence = build_controlled_gqa_model(config, 53)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, tensor in reference.state_dict().items():
        actual = model.state_dict()[f"decoder.{name}"]
        if ".attn.c_attn." not in name:
            assert torch.equal(actual, tensor)
        else:
            q, k, v = tensor.split(8, 0)
            pieces = [q]
            for part in (k, v):
                pieces.append(
                    torch.stack(
                        [
                            part[group * 8 // heads : (group + 1) * 8 // heads]
                            .reshape(4 // heads, 2, *part.shape[1:])
                            .mean(0)
                            for group in range(heads)
                        ]
                    ).reshape(heads * 2, *part.shape[1:])
                )
            assert torch.equal(actual, torch.cat(pieces, 0))
    assert "not full-QKV homology" in evidence["policy"]
    assert model.wte.weight is model.lm_head.weight
    if heads == 4:
        model.eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        torch.testing.assert_close(model(ids), reference(ids), rtol=0, atol=0)


def test_matched_actual_trials_replay_with_unequal_parameters_and_shared_exposure():
    controls = replace(default_gqa_plan(), seeds=(53, 54))
    rng = torch.get_rng_state().clone()
    report = run_gqa_comparison(controls, TRAIN, HELDOUT)
    assert verify_gqa_experiment_report(report) == controls
    replay_gqa_experiment_report(report, TRAIN, HELDOUT)
    assert torch.equal(torch.get_rng_state(), rng)
    for trial in report["trials"]:
        assert (
            len({arm["initialization"]["exact_shared_sha256"] for arm in trial["arms"]})
            == 1
        )
        assert (
            len(
                {arm["initialization"]["reference_mha_sha256"] for arm in trial["arms"]}
            )
            == 1
        )
        assert all(
            arm["initial_model_sha256"] != arm["final_model_sha256"]
            for arm in trial["arms"]
        )
        assert (
            len(
                {
                    sum(row["target_tokens"] for row in arm["trace"])
                    for arm in trial["arms"]
                }
            )
            == 1
        )


def test_complete_budget_rejects_before_model_allocation(monkeypatch):
    import ai_journey.gqa_experiment as experiment

    monkeypatch.setattr(
        experiment,
        "build_controlled_gqa_model",
        lambda *a: pytest.fail("no allocation"),
    )
    controls = replace(default_gqa_plan(), seeds=(1, 2, 3, 4), steps=16, batch_size=8)
    with pytest.raises(ValueError, match="work_units"):
        run_gqa_comparison(controls, TRAIN, b"x" * 8190 + b"\n")
