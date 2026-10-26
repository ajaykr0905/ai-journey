"""Exact MoE resume and adversarial restart controls are independent of reports."""

from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from ai_journey.moe_checkpoint import _digest, load_moe_checkpoint, save_moe_checkpoint
from ai_journey.moe_training import train_moe_step
from ai_journey.training_models import build_training_model, training_model_fingerprint
from test_moe_training import state

SOURCE = sha256(b"public checkpoint fixture").hexdigest()
CONTRACT = {
    "seed": 53,
    "balance_weight": 0.01,
    "z_weight": 0.0,
    "gradient_clip": 1.0,
    "batch_order": "fixed-token-v1",
}
INPUTS = torch.tensor([[1, 2, 3]])
TARGETS = torch.tensor([[2, 3, 4]])


def save(path, model, optimizer, step):
    save_moe_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=step,
        training_contract=CONTRACT,
        source_sha256=SOURCE,
    )


def load(path, model, optimizer, **kwargs):
    return load_moe_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_contract=CONTRACT,
        source_sha256=SOURCE,
        **kwargs,
    )


def test_split_training_restores_exact_model_optimizer_and_rng(tmp_path):
    torch.manual_seed(53)
    model, optimizer = state()
    train_moe_step(model, optimizer, INPUTS, TARGETS)
    path = tmp_path / "state.pt"
    save(path, model, optimizer, 1)
    uninterrupted = train_moe_step(model, optimizer, INPUTS, TARGETS)
    expected = training_model_fingerprint(model)
    expected_optimizer = _digest(optimizer.state_dict())
    restored, restored_optimizer = state()
    assert load(path, restored, restored_optimizer) == 1
    resumed = train_moe_step(restored, restored_optimizer, INPUTS, TARGETS)
    assert resumed == uninterrupted
    assert training_model_fingerprint(restored) == expected
    assert _digest(restored_optimizer.state_dict()) == expected_optimizer


@pytest.mark.parametrize(
    "corruption", ["seal", "counter", "moment_shape", "rng", "tensor_dtype", "controls"]
)
def test_corruption_is_rejected_without_changing_live_state(tmp_path, corruption):
    model, optimizer = state()
    train_moe_step(model, optimizer, INPUTS, TARGETS)
    path = tmp_path / "state.pt"
    save(path, model, optimizer, 1)
    payload = torch.load(path, weights_only=True)
    if corruption == "seal":
        payload["training_contract"]["seed"] = 99
    elif corruption == "counter":
        next(iter(payload["optimizer_state"]["state"].values()))["step"].fill_(0)
    elif corruption == "moment_shape":
        next(iter(payload["optimizer_state"]["state"].values()))["exp_avg"] = (
            torch.zeros(1)
        )
    elif corruption == "rng":
        payload["torch_rng_state"] = torch.zeros(3, dtype=torch.uint8)
    elif corruption == "tensor_dtype":
        payload["model_state"]["wte.weight"] = payload["model_state"][
            "wte.weight"
        ].double()
    else:
        payload["optimizer_state"]["param_groups"][0]["capturable"] = True
    if corruption != "seal":
        payload["checkpoint_sha256"] = _digest(
            {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
        )
    torch.save(payload, path)
    before, before_optimizer, rng = (
        training_model_fingerprint(model),
        _digest(optimizer.state_dict()),
        torch.get_rng_state().clone(),
    )
    model.ln_f.eval()
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in model.modules()]
    with pytest.raises((ValueError, RuntimeError)):
        load(path, model, optimizer)
    assert training_model_fingerprint(model) == before
    assert _digest(optimizer.state_dict()) == before_optimizer
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(parameter.grad, torch.ones_like(parameter))
    assert [module.training for module in model.modules()] == modes


def test_shape_compatible_capacity_change_is_rejected(tmp_path):
    model, optimizer = state()
    path = tmp_path / "state.pt"
    save(path, model, optimizer, 0)
    changed = build_training_model(replace(model.config, capacity_factor=2))
    from ai_journey.moe_training import build_moe_optimizer

    with pytest.raises(ValueError, match="architecture"):
        load(path, changed, build_moe_optimizer(changed, learning_rate=0.001))


def test_alias_equal_training_contract_is_not_accepted(tmp_path):
    model, optimizer = state()
    path = tmp_path / "state.pt"
    save(path, model, optimizer, 0)
    with pytest.raises(ValueError, match="training contract"):
        load_moe_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            training_contract={**CONTRACT, "seed": 53.0},
            source_sha256=SOURCE,
        )


def test_invalid_live_state_cannot_overwrite_checkpoint(tmp_path):
    model, optimizer = state()
    path = tmp_path / "state.pt"
    save(path, model, optimizer, 0)
    before = path.read_bytes()
    optimizer.param_groups[0]["capturable"] = True
    with pytest.raises(ValueError):
        save(path, model, optimizer, 0)
    assert path.read_bytes() == before


def test_failed_atomic_write_preserves_durable_checkpoint_and_cleans_temporary(
    tmp_path, monkeypatch
):
    model, optimizer = state()
    path = tmp_path / "state.pt"
    save(path, model, optimizer, 0)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected serialization failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="injected"):
        save(path, model, optimizer, 0)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_checkpoint_archive_byte_budget_rejects_before_deserialization(
    tmp_path, monkeypatch
):
    model, optimizer = state()
    path = tmp_path / "state.pt"
    save(path, model, optimizer, 0)
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("must not deserialize")
    )
    with pytest.raises(ValueError, match="bounded weights-only"):
        load(path, model, optimizer, max_checkpoint_bytes=1)
