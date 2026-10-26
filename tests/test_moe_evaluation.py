from dataclasses import replace
import math

import pytest
import torch

from ai_journey.ddp_evaluation import Document
from ai_journey.decoder_variant_evaluation import evaluate_decoder_documents
from ai_journey.moe_evaluation import evaluate_moe_documents
from ai_journey.training_models import build_training_model, training_model_fingerprint
from test_moe_experiment_plan import plan

DOCUMENTS = (Document("a", b"an authored public example"), Document("b", b"tail!"))


def model(*, dense=False):
    controls = plan()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(53)
        return build_training_model(controls.arms[0 if dense else 1].config)


def test_complete_dropless_evaluation_covers_each_target_and_excludes_padding():
    network = model()
    result = evaluate_moe_documents(network, DOCUMENTS, batch_size=3)
    assert result["capacity_policy"] == "dropless_eval"
    assert result["target_tokens"] == sum(len(doc.content) - 1 for doc in DOCUMENTS)
    assert result["nll_sum"] == math.fsum(
        value for batch in result["batches"] for value in batch["token_nll"]
    )
    summary = result["routing_summary"][0]
    assert summary["valid_tokens"] == result["target_tokens"]
    assert sum(summary["accepted_counts"]) == 2 * result["target_tokens"]
    assert sum(summary["dropped_counts"]) == 0
    for batch in result["batches"]:
        for row in batch["routing"]:
            assert row["capacity_applied"] is False
            for counts, mask in zip(
                row["accepted_per_token"], batch["token_mask"], strict=True
            ):
                assert counts == [2 if valid else 0 for valid in mask]


def test_dense_control_matches_existing_complete_evaluator():
    network = model(dense=True)
    actual = evaluate_moe_documents(network, DOCUMENTS, batch_size=3)
    expected = evaluate_decoder_documents(network, DOCUMENTS, batch_size=3)
    for field in (
        "target_tokens",
        "nll_sum",
        "mean_nll",
        "bits_per_byte",
        "perplexity",
    ):
        assert actual[field] == expected[field]
    assert actual["routing_summary"] == []


def test_actual_model_batching_changes_only_recorded_grouping_with_close_scores():
    network = model()
    one = evaluate_moe_documents(network, DOCUMENTS, batch_size=1)
    many = evaluate_moe_documents(network, DOCUMENTS, batch_size=3)
    assert one["target_tokens"] == many["target_tokens"]
    assert one["mean_nll"] == pytest.approx(many["mean_nll"], rel=1e-6)
    assert one["batch_schedule_sha256"] != many["batch_schedule_sha256"]


def test_evaluation_restores_rng_modes_parameters_and_existing_gradients():
    network = model().train()
    network.h[0].ln_1.eval()
    parameter = next(network.parameters())
    parameter.grad = torch.ones_like(parameter)
    gradient = parameter.grad.clone()
    identity = training_model_fingerprint(network)
    modes = [(module, module.training) for module in network.modules()]
    rng = torch.get_rng_state().clone()
    evaluate_moe_documents(network, DOCUMENTS)
    assert training_model_fingerprint(network) == identity
    assert all(module.training == mode for module, mode in modes)
    assert torch.equal(parameter.grad, gradient)
    assert torch.equal(torch.get_rng_state(), rng)


def test_nonfinite_model_restores_modes_on_failure():
    network = model().train()
    next(network.parameters()).data.fill_(float("nan"))
    with pytest.raises((ValueError, FloatingPointError), match="finite"):
        evaluate_moe_documents(network, DOCUMENTS)
    assert network.training


@pytest.mark.parametrize(
    "documents,batch",
    [
        ((), 2),
        ((DOCUMENTS[0],) * 2, 2),
        (DOCUMENTS, True),
        (DOCUMENTS, 17),
        ((Document("x", b"x" * 8194),), 1),
    ],
)
def test_evaluation_rejects_invalid_or_unbounded_work(documents, batch):
    with pytest.raises(ValueError):
        evaluate_moe_documents(model(), documents, batch_size=batch)


def test_non_float32_evaluation_is_not_mislabeled_as_pinned_cpu_protocol():
    with pytest.raises(ValueError, match="float32"):
        evaluate_moe_documents(model().double(), DOCUMENTS)


def test_normal_evaluation_never_enables_configured_capacity_dropping():
    network = model()
    network.config = replace(network.config, capacity_factor=0.01)
    for block in network.h:
        if hasattr(block.mlp, "router"):
            block.mlp.config = network.config
            block.mlp.router.projection.weight.data.zero_()
    result = evaluate_moe_documents(network, DOCUMENTS)
    assert sum(result["routing_summary"][0]["dropped_counts"]) == 0
