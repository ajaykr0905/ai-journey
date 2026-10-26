"""Inference artifacts must validate the complete restart and source contract."""

from hashlib import sha256

import pytest
import torch

from ai_journey.moe_checkpoint import _digest, save_moe_checkpoint
from ai_journey.moe_inference import load_moe_model_checkpoint
from ai_journey.moe_training import train_moe_step
from ai_journey.training_models import training_model_fingerprint
from test_moe_checkpoint import CONTRACT, INPUTS, TARGETS
from test_moe_training import state

SOURCE_BYTES = b"public inference fixture"


def artifact(tmp_path, *, double=False):
    model, optimizer = state()
    if double:
        model.double()
    train_moe_step(model, optimizer, INPUTS, TARGETS)
    path = tmp_path / "moe.pt"
    save_moe_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        training_contract=CONTRACT,
        source_sha256=sha256(SOURCE_BYTES).hexdigest(),
    )
    return path, model


@pytest.mark.parametrize("double", [False, True])
def test_restart_valid_artifact_has_exact_frozen_eval_logits_and_preserves_rng(
    tmp_path, double
):
    path, trained = artifact(tmp_path, double=double)
    trained.eval()
    rng = torch.get_rng_state().clone()
    frozen = load_moe_model_checkpoint(
        path, config=trained.config, training_contract=CONTRACT, source=SOURCE_BYTES
    )
    assert torch.equal(torch.get_rng_state(), rng)
    assert training_model_fingerprint(frozen) == training_model_fingerprint(trained)
    assert not any(parameter.requires_grad for parameter in frozen.parameters())
    assert not any(module.training for module in frozen.modules())
    assert frozen.lm_head.weight is frozen.wte.weight
    with torch.inference_mode():
        torch.testing.assert_close(frozen(INPUTS), trained(INPUTS), rtol=0, atol=0)
        assert all(
            not stats["capacity_applied"]
            for stats in frozen.forward_training(INPUTS).routing
        )


def test_wrong_source_is_rejected_before_model_allocation(tmp_path, monkeypatch):
    path, trained = artifact(tmp_path)
    monkeypatch.setattr(
        "ai_journey.moe_inference.build_training_model",
        lambda *a, **k: pytest.fail("must not allocate"),
    )
    with pytest.raises(ValueError, match="source bytes"):
        load_moe_model_checkpoint(
            path,
            config=trained.config,
            training_contract=CONTRACT,
            source=b"wrong public bytes",
        )


def test_resealed_missing_optimizer_parameter_is_not_inference_ready(tmp_path):
    path, trained = artifact(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload["optimizer_contract"][0]["names"].pop()
    payload["checkpoint_sha256"] = _digest(
        {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
    )
    torch.save(payload, path)
    with pytest.raises(ValueError, match="every parameter"):
        load_moe_model_checkpoint(
            path, config=trained.config, training_contract=CONTRACT, source=SOURCE_BYTES
        )


def test_inference_checkpoint_retains_parameter_admission(tmp_path, monkeypatch):
    path, trained = artifact(tmp_path)
    monkeypatch.setattr(
        "ai_journey.training_models.MoEDecoderModel",
        lambda *a, **k: pytest.fail("must not allocate"),
    )
    with pytest.raises(ValueError, match="parameter budget"):
        load_moe_model_checkpoint(
            path,
            config=trained.config,
            training_contract=CONTRACT,
            source=SOURCE_BYTES,
            max_parameters=1,
        )
