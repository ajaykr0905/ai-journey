"""Serialized moments must be checked before AdamW silently casts their state."""

import copy

import pytest
import torch

from ai_journey.ddp_acceptance import exact_state
from ai_journey.ddp_checkpoint import load_checkpoint, save_checkpoint
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from ai_journey.moe_checkpoint import _digest, load_moe_checkpoint, save_moe_checkpoint
from test_ddp_checkpoint import setup_state
from test_moe_checkpoint import CONTRACT, SOURCE, INPUTS, TARGETS
from test_moe_training import state as moe_state
from ai_journey.moe_training import train_moe_step
from ai_journey.training_optimizer_state import validate_serialized_adamw_state


def test_distributed_restart_rejects_moment_precision_coercion(tmp_path):
    model, optimizer, config = setup_state()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    path = tmp_path / "state.pt"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=1,
        )
        payload = torch.load(path, weights_only=True)
        moment = next(iter(payload["optimizer"]["state"].values()))
        moment["exp_avg"] = moment["exp_avg"].double()
        torch.save(payload, path)
        before = copy.deepcopy(optimizer.state_dict())
        with pytest.raises(RuntimeError, match="restoration"):
            load_checkpoint(
                path, model=model, optimizer=optimizer, config=config, source=b"public"
            )
        assert exact_state(optimizer.state_dict(), before)


def test_expert_restart_rejects_moment_precision_coercion(tmp_path):
    model, optimizer = moe_state()
    train_moe_step(model, optimizer, INPUTS, TARGETS)
    path = tmp_path / "state.pt"
    save_moe_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        training_contract=CONTRACT,
        source_sha256=SOURCE,
    )
    payload = torch.load(path, weights_only=True)
    moment = next(iter(payload["optimizer_state"]["state"].values()))
    moment["exp_avg"] = moment["exp_avg"].double()
    payload["checkpoint_sha256"] = _digest(
        {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
    )
    torch.save(payload, path)
    before = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError, match="serialized AdamW"):
        load_moe_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            training_contract=CONTRACT,
            source_sha256=SOURCE,
        )
    assert exact_state(optimizer.state_dict(), before)


def test_tiny_archive_cannot_cast_a_billion_element_optimizer_view(
    tmp_path, monkeypatch
):
    model, optimizer, config = setup_state()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    path = tmp_path / "state.pt"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=1,
        )
        payload = torch.load(path, weights_only=True)
        moment = next(iter(payload["optimizer"]["state"].values()))
        moment["exp_avg"] = torch.ones(1, dtype=torch.float64).expand(2**30)
        torch.save(payload, path)
        assert path.stat().st_size < 1024 * 1024
        original = torch.Tensor.to

        def bounded_to(tensor, *args, **kwargs):
            if tensor.numel() >= 2**30:
                pytest.fail("admission must precede optimizer tensor conversion")
            return original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", bounded_to)
        with pytest.raises(RuntimeError, match="validation|restoration"):
            load_checkpoint(
                path, model=model, optimizer=optimizer, config=config, source=b"public"
            )


@pytest.mark.parametrize(
    "mutation",
    ["dtype", "shape", "field", "negative", "binding", "bool_id", "unknown", "counter"],
)
def test_raw_optimizer_schema_rejects_before_any_loader_coercion(mutation):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    raw = copy.deepcopy(optimizer.state_dict())
    moment = raw["state"][0]
    if mutation == "dtype":
        moment["exp_avg"] = moment["exp_avg"].double()
    elif mutation == "shape":
        moment["exp_avg"] = torch.ones(1)
    elif mutation == "field":
        moment["undeclared"] = torch.ones(1)
    elif mutation == "negative":
        moment["exp_avg_sq"].fill_(-1)
    elif mutation == "binding":
        raw["param_groups"][0]["params"].reverse()
    elif mutation == "bool_id":
        raw["param_groups"][0]["params"][0] = False
    elif mutation == "unknown":
        raw["state"][999] = copy.deepcopy(moment)
    else:
        moment["step"] = torch.tensor(True)
    with pytest.raises(ValueError, match="serialized AdamW"):
        validate_serialized_adamw_state(model, optimizer, raw, step=1)


@pytest.mark.parametrize("amsgrad", [False, True])
def test_valid_raw_adamw_state_retains_its_full_precision_and_next_update(amsgrad):
    model = torch.nn.Linear(2, 2).double()
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=amsgrad)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    raw = copy.deepcopy(optimizer.state_dict())
    validate_serialized_adamw_state(model, optimizer, raw, step=1)
    optimizer.load_state_dict(raw)
    optimizer.step()
    validate_serialized_adamw_state(model, optimizer, optimizer.state_dict(), step=2)
