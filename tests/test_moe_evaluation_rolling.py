import copy
import math

import pytest
import torch
from torch.nn import functional as F

from ai_journey.ddp_evaluation import Document
from ai_journey.ddp_generation import generate_tokens, SamplingConfig
from ai_journey.decoder_variant_comparison import decoder_comparison_digest
from ai_journey.moe_cli import main
from ai_journey.moe_evaluation import evaluate_moe_documents
from ai_journey.moe_evaluation_rolling import (
    evaluate_moe_rolling_documents,
    replay_moe_rolling_report,
    verify_moe_rolling_report,
)
from ai_journey.moe_experiment import build_matched_moe_experiment_model
from ai_journey.moe_experiment_artifacts import publish_moe_experiment
from test_moe_experiment import TRAIN, HELDOUT
from test_moe_experiment_plan import plan

DOCUMENTS = (Document("long", b"abcdefghijk"), Document("short", b"xy"))


@pytest.mark.parametrize("sparse", [False, True])
def test_every_target_matches_brute_force_last_prefix_and_actual_serving_policy(sparse):
    model = build_matched_moe_experiment_model(
        plan().arms[int(sparse)].config, 53
    ).eval()
    report = evaluate_moe_rolling_documents(model, DOCUMENTS, batch_size=3)
    expected = []
    with torch.inference_mode():
        for document in DOCUMENTS:
            for offset in range(1, len(document.content)):
                prefix = document.content[max(0, offset - 4) : offset]
                logits = model(torch.tensor([list(prefix)]))[:, -1]
                expected.append(
                    float(
                        F.cross_entropy(
                            logits, torch.tensor([document.content[offset]])
                        )
                    )
                )
                generated = generate_tokens(
                    model, prefix, SamplingConfig(max_new_tokens=1, temperature=0)
                )
                assert generated["tokens"] == [int(logits.argmax())]
    actual = [value for batch in report["batches"] for value in batch["target_nll"]]
    assert actual == pytest.approx(expected, abs=2e-6)
    assert report["target_tokens"] == 11
    assert report["context_tokens"] == sum(min(4, i) for i in range(1, 11)) + 1
    assert report["mean_nll"] == pytest.approx(math.fsum(expected) / 11, abs=2e-6)
    rows = [row for batch in report["batches"] for row in batch["rows"]]
    assert rows[4] == {
        "document_index": 0,
        "target_offset": 5,
        "context_start": 1,
        "context_tokens": 4,
    }
    assert rows[-1]["context_start"] == 0 and rows[-1]["document_index"] == 1
    assert report["mean_nll"] != evaluate_moe_documents(model, DOCUMENTS)["mean_nll"]
    if sparse:
        summary = report["routing_summary"][0]
        assert (
            summary["valid_tokens"]
            == report["context_tokens"]
            > report["target_tokens"]
        )
        assert sum(summary["attempted_counts"]) == 2 * report["context_tokens"]
        assert sum(summary["dropped_counts"]) == 0


def test_rolling_preserves_mixed_modes_rng_gradients_and_weights_and_replays():
    model = build_matched_moe_experiment_model(plan().arms[1].config, 53)
    model.train()
    model.h[0].attn.eval()
    next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
    modes = [module.training for module in model.modules()]
    state = copy.deepcopy(model.state_dict())
    gradient = next(model.parameters()).grad.clone()
    rng = torch.get_rng_state().clone()
    report = evaluate_moe_rolling_documents(model, DOCUMENTS)
    replay_moe_rolling_report(report, model, DOCUMENTS)
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(next(model.parameters()).grad, gradient)
    assert all(
        torch.equal(value, state[name]) for name, value in model.state_dict().items()
    )


@pytest.fixture
def measured():
    model = build_matched_moe_experiment_model(plan().arms[1].config, 53)
    return model, evaluate_moe_rolling_documents(model, DOCUMENTS)


@pytest.mark.parametrize(
    "mutation",
    [
        "skip",
        "repeat",
        "context",
        "score_only_mask",
        "mean",
        "alias",
        "resource",
        "routing",
        "summary",
        "length",
    ],
)
def test_resealed_impossible_rolling_observations_are_rejected(measured, mutation):
    _, report = measured
    report = copy.deepcopy(report)
    if mutation == "skip":
        report["batches"].pop()
    elif mutation == "repeat":
        report["batches"][0]["rows"][1] = report["batches"][0]["rows"][0]
    elif mutation == "context":
        report["batches"][1]["rows"][0]["context_start"] = 0
    elif mutation == "score_only_mask":
        report["batches"][0]["context_mask"][1][0] = False
    elif mutation == "mean":
        report["mean_nll"] += 0.01
    elif mutation == "alias":
        report["runtime"]["threads"] = True
    elif mutation == "resource":
        report["resource_admission"]["work_units"] -= 1
    elif mutation == "routing":
        report["batches"][0]["routing"][0]["capacity_applied"] = True
    elif mutation == "summary":
        report["routing_summary"][0]["valid_tokens"] -= 1
    else:
        report["documents"][0]["bytes"] += 1
    report["report_sha256"] = decoder_comparison_digest(report)
    with pytest.raises(ValueError):
        verify_moe_rolling_report(report)


def test_consistent_fabricated_losses_pass_internal_check_but_fail_actual_replay(
    measured,
):
    model, report = measured
    report = copy.deepcopy(report)
    report["batches"][0]["target_nll"][0] += 1
    total = math.fsum(
        value for batch in report["batches"] for value in batch["target_nll"]
    )
    mean = total / report["target_tokens"]
    report.update(
        nll_sum=total,
        mean_nll=mean,
        bits_per_byte=mean / math.log(2),
        perplexity=math.exp(mean),
    )
    report["report_sha256"] = decoder_comparison_digest(report)
    verify_moe_rolling_report(report)
    with pytest.raises(ValueError, match="actual execution"):
        replay_moe_rolling_report(report, model, DOCUMENTS)


def test_replay_rejects_wrong_model_or_source_before_execution(measured, monkeypatch):
    model, report = measured
    monkeypatch.setattr(
        model, "forward_training", lambda *a, **kw: pytest.fail("no forward")
    )
    with pytest.raises(ValueError, match="recorded CPU runtime"):
        replay_moe_rolling_report(report, model, (Document("other", b"different"),))
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(ValueError, match="recorded CPU runtime"):
        replay_moe_rolling_report(report, model, DOCUMENTS)


@pytest.mark.parametrize(
    "documents,batch",
    [
        ((), 4),
        ((DOCUMENTS[0], DOCUMENTS[0]), 4),
        ((Document("big", b"x" * 8194),), 4),
        (DOCUMENTS, True),
        (DOCUMENTS, 17),
    ],
)
def test_invalid_or_excessive_work_is_rejected_before_forward(
    documents, batch, monkeypatch
):
    model = build_matched_moe_experiment_model(plan().arms[1].config, 53)
    monkeypatch.setattr(
        model, "forward_training", lambda *a, **kw: pytest.fail("no forward")
    )
    with pytest.raises(ValueError):
        evaluate_moe_rolling_documents(model, documents, batch_size=batch)


def test_nonfinite_loss_and_float64_model_are_not_published(monkeypatch):
    model = build_matched_moe_experiment_model(plan().arms[0].config, 53)
    monkeypatch.setattr(
        model, "forward", lambda ids: torch.full((*ids.shape, 256), float("nan"))
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        evaluate_moe_rolling_documents(model, DOCUMENTS)
    model.double()
    with pytest.raises(ValueError, match="float32 CPU"):
        evaluate_moe_rolling_documents(model, DOCUMENTS)


def test_cli_rolling_generation_internal_verification_and_actual_replay(
    tmp_path, capsys
):
    destination = tmp_path / "trained"
    publish_moe_experiment(plan(), TRAIN, HELDOUT, destination, arm="top2", seed=53)
    train, heldout, output = (
        tmp_path / name for name in ("train", "heldout", "rolling.json")
    )
    train.write_bytes(TRAIN)
    heldout.write_bytes(HELDOUT)
    arguments = [
        "--directory",
        str(destination),
        "--train",
        str(train),
        "--heldout",
        str(heldout),
    ]
    assert main(["rolling", *arguments, "--output", str(output)]) == 0
    original = output.read_bytes()
    assert main(["rolling-verify", "--report", str(output)]) == 0
    assert main(["rolling-replay", *arguments, "--report", str(output)]) == 0
    assert output.read_bytes() == original
    with pytest.raises(SystemExit):
        main(["rolling", *arguments, "--output", str(output)])
    heldout.write_bytes(TRAIN)
    with pytest.raises(SystemExit):
        main(["rolling-replay", *arguments, "--report", str(output)])
    assert "rolling_replayed=" in capsys.readouterr().out
