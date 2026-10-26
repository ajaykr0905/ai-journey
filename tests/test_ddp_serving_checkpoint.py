from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from ai_journey import ddp_pipeline
from ai_journey.gpt2_training import (
    build_gpt2_optimizer,
    gpt2_model_fingerprint,
    training_spec_from_mapping,
)
from ai_journey.training_models import build_training_model, model_config_mapping


def fixture():
    source = b"independently authored public test fixture"
    mapping = json.loads(
        (Path(__file__).resolve().parents[1] / "config/gpt2-training.json").read_text()
    )
    mapping["source"]["sha256"] = sha256(source).hexdigest()
    spec = training_spec_from_mapping(mapping)
    with torch.random.fork_rng(devices=[]):
        model = build_training_model(spec.model).double()
    payload = {
        "schema_version": 1,
        "world_size": 1,
        "step": 0,
        "model_config": model_config_mapping(spec.model),
        "training_config": asdict(spec.training),
        "source_sha256": sha256(source).hexdigest(),
        "model_sha256": gpt2_model_fingerprint(model),
        "model": model.state_dict(),
        "optimizer": build_gpt2_optimizer(model, spec.training).state_dict(),
        "rank_rngs": [torch.get_rng_state()],
    }
    return source, spec, payload


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(schema_version=True),
        lambda p: p.update(schema_version=2),
        lambda p: p.update(extra="untrusted"),
        lambda p: p.pop("step"),
        lambda p: p.update(step=True),
        lambda p: p.update(step=-1),
        lambda p: p.update(step=999),
        lambda p: p.update(world_size=True),
        lambda p: p.update(world_size=999),
        lambda p: p.update(rank_rngs=[]),
        lambda p: p.update(rank_rngs=[torch.zeros(1)]),
        lambda p: p.update(model_sha256="bad"),
    ],
)
def test_invalid_serving_envelopes_are_rejected_before_model_allocation(
    tmp_path, monkeypatch, change
):
    source, spec, payload = fixture()
    change(payload)
    path = tmp_path / "state.pt"
    torch.save(payload, path)

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid envelope must not allocate a model")

    monkeypatch.setattr(ddp_pipeline, "build_training_model", unexpected)
    with pytest.raises(ValueError):
        ddp_pipeline.load_serving_model(path, spec, source)


def test_serving_loader_binds_source_to_declared_spec_not_just_checkpoint(tmp_path):
    source, spec, payload = fixture()
    changed = source + b" changed"
    payload["source_sha256"] = sha256(changed).hexdigest()
    path = tmp_path / "state.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="provenance"):
        ddp_pipeline.load_serving_model(path, spec, changed)


@pytest.mark.parametrize("mutation", ["dtype", "weight_decay", "maximize"])
def test_serving_loader_rejects_cast_tensor_or_undeclared_optimizer_controls(
    tmp_path, mutation
):
    source, spec, payload = fixture()
    if mutation == "dtype":
        payload["model"] = {
            name: value.float() for name, value in payload["model"].items()
        }
    elif mutation == "weight_decay":
        payload["optimizer"]["param_groups"][0]["weight_decay"] += 0.1
    else:
        payload["optimizer"]["param_groups"][0]["maximize"] = True
    path = tmp_path / "state.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        ddp_pipeline.load_serving_model(path, spec, source)


def test_valid_serving_checkpoint_preserves_caller_rng_and_freezes_model(tmp_path):
    source, spec, payload = fixture()
    path = tmp_path / "state.pt"
    torch.save(payload, path)
    before = torch.get_rng_state().clone()
    loaded = ddp_pipeline.load_serving_model(path, spec, source)
    assert torch.equal(before, torch.get_rng_state())
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
