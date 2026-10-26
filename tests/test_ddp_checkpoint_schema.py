"""Restart must reject coerced contracts and tensor schemas before live loading."""

import copy

import pytest
import torch

from ai_journey.ddp_acceptance import exact_state
from ai_journey.ddp_checkpoint import load_checkpoint, save_checkpoint
from ai_journey.ddp_contracts import Topology
from ai_journey.ddp_runtime import process_group
from test_ddp_checkpoint import setup_state


@pytest.mark.parametrize("corruption", ["world_size", "model", "schedule", "dtype"])
def test_inexact_checkpoint_schema_rejects_before_loading_live_model(
    tmp_path, monkeypatch, corruption
):
    model, optimizer, config = setup_state()
    path = tmp_path / "state.pt"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=0,
        )
        payload = torch.load(path, weights_only=True)
        if corruption == "world_size":
            payload["world_size"] = True
        elif corruption == "model":
            payload["model_config"]["layer_count"] = True
        elif corruption == "schedule":
            payload["training_config"]["warmup_steps"] = False
        else:
            payload["model"]["wte.weight"] = payload["model"]["wte.weight"].double()
        torch.save(payload, path)
        before_model = copy.deepcopy(model.state_dict())
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        before_rng = torch.get_rng_state().clone()
        monkeypatch.setattr(
            model,
            "load_state_dict",
            lambda *a, **k: pytest.fail("schema admission must precede model mutation"),
        )
        with pytest.raises(RuntimeError, match="checkpoint validation"):
            load_checkpoint(
                path, model=model, optimizer=optimizer, config=config, source=b"public"
            )
        assert exact_state(model.state_dict(), before_model)
        assert exact_state(optimizer.state_dict(), before_optimizer)
        assert torch.equal(torch.get_rng_state(), before_rng)


def test_valid_float64_schema_restores_without_precision_coercion(tmp_path):
    model, optimizer, config = setup_state()
    model.double()
    path = tmp_path / "double.pt"
    with process_group(Topology(0, 1), rendezvous=(tmp_path / "rdzv").as_uri()):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            source=b"public",
            step=0,
        )
        before = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            model.wte.weight.add_(1)
        assert (
            load_checkpoint(
                path, model=model, optimizer=optimizer, config=config, source=b"public"
            )
            == 0
        )
        assert exact_state(model.state_dict(), before)
