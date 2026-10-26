"""Adversarial restart contracts for the controlled training entry point."""

import copy

import pytest
import torch

from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_training import (
    GPT2BatchCursor,
    GPT2ByteCorpus,
    GPT2TrainingConfig,
    build_gpt2_optimizer,
    load_training_checkpoint,
    save_training_checkpoint,
    train_optimizer_step,
)


@pytest.fixture
def restart_state(tmp_path):
    model = GPT2LanguageModel(GPT2Config(256, 4, 8, 2, 1))
    config = GPT2TrainingConfig(2, 1, 1, 4, 0.01, 0.001, 0, 0.1, 1.0, 52)
    corpus = GPT2ByteCorpus.from_bytes(b"public restart fixture" * 3)
    cursor = GPT2BatchCursor(corpus, config)
    optimizer = build_gpt2_optimizer(model, config)
    train_optimizer_step(model, cursor, optimizer, config, step=0)
    path = tmp_path / "checkpoint.pt"
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        cursor=cursor,
        config=config,
        corpus_fingerprint=corpus.fingerprint(),
    )
    save_training_checkpoint(path, step=1, **kwargs)
    return path, kwargs


@pytest.mark.parametrize("corruption", ["shape", "negative_variance", "counter", "lr"])
def test_invalid_finite_adamw_state_is_rejected_transactionally(
    restart_state, corruption
):
    path, kwargs = restart_state
    payload = torch.load(path, weights_only=True)
    state = next(iter(payload["optimizer_state"]["state"].values()))
    if corruption == "shape":
        state["exp_avg"] = torch.zeros(1)
    elif corruption == "negative_variance":
        state["exp_avg_sq"].fill_(-1)
    elif corruption == "counter":
        state["step"].fill_(0)
    else:
        payload["optimizer_state"]["param_groups"][0]["lr"] = -0.1
    torch.save(payload, path)
    before = copy.deepcopy(kwargs["model"].state_dict())
    cursor = kwargs["cursor"].state_dict()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="AdamW"):
        load_training_checkpoint(path, **kwargs)
    assert all(
        torch.equal(tensor, before[name])
        for name, tensor in kwargs["model"].state_dict().items()
    )
    assert kwargs["cursor"].state_dict() == cursor
    assert torch.equal(torch.get_rng_state(), rng)
    for state in kwargs["optimizer"].state.values():
        assert state["step"].item() == 1
        assert (state["exp_avg_sq"] >= 0).all()


def test_invalid_live_adamw_state_cannot_replace_durable_checkpoint(restart_state):
    path, kwargs = restart_state
    before = path.read_bytes()
    next(iter(kwargs["optimizer"].state.values()))["exp_avg_sq"].fill_(-1)
    with pytest.raises(ValueError, match="AdamW"):
        save_training_checkpoint(path, step=1, **kwargs)
    assert path.read_bytes() == before
