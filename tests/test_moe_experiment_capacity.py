from dataclasses import replace
import math

import pytest
import torch

from ai_journey.ddp_evaluation import Document
from ai_journey.moe_experiment_capacity import probe_moe_training_capacity
from ai_journey.training_models import build_training_model, training_model_fingerprint
from test_moe_experiment_plan import plan


def skewed_model():
    config = replace(plan().arms[1].config, capacity_factor=0.01)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(53)
        model = build_training_model(config)
    for block in model.h:
        if hasattr(block.mlp, "router"):
            block.mlp.router.projection.weight.data.zero_()
    return model


def test_real_capacity_probe_records_batch_dependent_admission_not_inference():
    model = skewed_model()
    documents = (Document("public", b"public capacity pressure control"),)
    result = probe_moe_training_capacity(model, documents, batch_sizes=(1, 4))
    assert "training-mode" in result["scope"]
    assert sum(result["dropless"]["routing_summary"][0]["dropped_counts"]) == 0
    accepted = []
    for case in result["cases"]:
        summary = case["routing_summary"][0]
        assert sum(summary["dropped_counts"]) > 0
        accepted.append(sum(summary["accepted_counts"]))
        strata = case["loss_strata"][0]
        assert strata["target_tokens"] == case["target_tokens"]
        assert (
            sum(row["target_tokens"] for row in strata["strata"])
            == case["target_tokens"]
        )
        assert math.fsum(row["nll_sum"] for row in strata["strata"]) == pytest.approx(
            case["nll_sum"]
        )
    assert accepted[0] != accepted[1]
    assert result == probe_moe_training_capacity(model, documents, batch_sizes=(1, 4))


def test_probe_preserves_weights_gradients_rng_and_mixed_modes():
    model = skewed_model().eval()
    model.h[0].ln_1.train()
    modes = [(module, module.training) for module in model.modules()]
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    before, rng = training_model_fingerprint(model), torch.get_rng_state().clone()
    probe_moe_training_capacity(model, (Document("public", b"public diagnostic"),))
    assert training_model_fingerprint(model) == before
    assert all(module.training == mode for module, mode in modes)
    assert torch.equal(parameter.grad, torch.ones_like(parameter))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("batches", [(True,), (0,), (9,), (1, 1), (), [1, 2]])
def test_invalid_probe_controls_are_rejected(batches):
    with pytest.raises(ValueError):
        probe_moe_training_capacity(
            skewed_model(),
            (Document("public", b"public diagnostic"),),
            batch_sizes=batches,
        )


def test_probe_refuses_dropout_confound_and_oversized_source():
    model = skewed_model()
    model.config = replace(
        model.config, gpt2=replace(model.config.gpt2, residual_dropout=0.1)
    )
    with pytest.raises(ValueError, match="zero-dropout"):
        probe_moe_training_capacity(model, (Document("p", b"public diagnostic"),))
    with pytest.raises(ValueError, match="bounded"):
        probe_moe_training_capacity(skewed_model(), (Document("p", b"p" * 8194),))
