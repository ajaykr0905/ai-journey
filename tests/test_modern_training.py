"""Modern decoder components use the same controlled restartable training path."""

from dataclasses import replace
from hashlib import sha256
from itertools import product

import pytest
import torch

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_training import (
    GPT2TrainingConfig,
    GPT2TrainingSource,
    GPT2TrainingSpec,
    run_training_experiment,
    training_spec_from_mapping,
)
from ai_journey.gpt2_training_evidence import (
    build_training_evidence,
    verify_training_replay,
)
from ai_journey.training_artifact import load_training_model_checkpoint
from ai_journey.training_models import training_model_fingerprint


def modern_spec(norm="rmsnorm", mlp="swiglu", position="rope"):
    source = b"public architecture training fixture\n" * 4
    base = GPT2Config(
        256,
        4,
        8,
        2,
        1,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    config = DecoderVariantConfig(base, norm, mlp, position)
    training = GPT2TrainingConfig(2, 1, 2, 4, 0.01, 0.001, 0, 0.1, 1.0, 52)
    provenance = GPT2TrainingSource(
        "Public fixture",
        "https://example.org/public",
        "fixture-v1",
        "CC0-1.0",
        sha256(source).hexdigest(),
    )
    return source, GPT2TrainingSpec(config, training, provenance)


@pytest.mark.parametrize(
    "norm,mlp,position",
    product(("layernorm", "rmsnorm"), ("gelu", "swiglu"), ("learned", "rope")),
)
def test_modern_training_restores_exact_state_across_all_component_choices(
    tmp_path, norm, mlp, position
):
    source, spec = modern_spec(norm, mlp, position)
    assert training_spec_from_mapping(spec.to_mapping()) == spec
    assert spec.to_mapping()["schema_version"] == 2
    full = run_training_experiment(source, spec, checkpoint_path=tmp_path / "full.pt")
    first = run_training_experiment(
        source, spec, checkpoint_path=tmp_path / "split.pt", step_count=1
    )
    resumed = run_training_experiment(
        source, spec, checkpoint_path=tmp_path / "split.pt", resume=True
    )
    assert first.completed_step == resumed.started_step == 1
    assert resumed.final_model_fingerprint == full.final_model_fingerprint
    assert resumed.metrics == full.metrics[1:]
    assert resumed.optimizer_groups.total_parameters == full.parameter_count
    full_state = torch.load(tmp_path / "full.pt", weights_only=True)
    split_state = torch.load(tmp_path / "split.pt", weights_only=True)
    for key, expected in full_state["model_state"].items():
        assert torch.equal(expected, split_state["model_state"][key])


def test_modern_segment_evidence_replays_including_its_prefix(tmp_path):
    source, spec = modern_spec()
    checkpoint = tmp_path / "state.pt"
    run_training_experiment(source, spec, checkpoint_path=checkpoint, step_count=1)
    run = run_training_experiment(source, spec, checkpoint_path=checkpoint, resume=True)
    evidence = build_training_evidence(run)
    assert verify_training_replay(source, evidence) == evidence


def test_checkpoint_rejects_different_rotary_contract_with_identical_shapes(tmp_path):
    source, spec = modern_spec()
    path = tmp_path / "state.pt"
    run_training_experiment(source, spec, checkpoint_path=path, step_count=1)
    original = path.read_bytes()
    changed = replace(spec, model=replace(spec.model, rope_base=1000))
    with pytest.raises(ValueError, match="model configuration"):
        run_training_experiment(source, changed, checkpoint_path=path, resume=True)
    assert path.read_bytes() == original


@pytest.mark.parametrize("modern", [False, True])
def test_verified_checkpoint_becomes_frozen_inference_model_without_rng_effects(
    tmp_path, modern
):
    source, spec = modern_spec()
    if not modern:
        spec = replace(spec, model=spec.model.gpt2)
    path = tmp_path / "state.pt"
    run = run_training_experiment(source, spec, checkpoint_path=path)
    rng = torch.get_rng_state().clone()
    model = load_training_model_checkpoint(path, spec=spec, source=source)
    assert torch.equal(torch.get_rng_state(), rng)
    assert training_model_fingerprint(model) == run.final_model_fingerprint
    assert not any(module.training for module in model.modules())
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert model.lm_head.weight is model.wte.weight
    with torch.inference_mode():
        assert torch.isfinite(model(torch.tensor([[1, 2, 3]]))).all()


def test_inference_loader_rejects_wrong_source_before_allocation(tmp_path, monkeypatch):
    source, spec = modern_spec()

    def unexpected(*args, **kwargs):
        raise AssertionError("allocation must not occur")

    monkeypatch.setattr("ai_journey.training_artifact.build_training_model", unexpected)
    with pytest.raises(ValueError, match="source bytes"):
        load_training_model_checkpoint(
            tmp_path / "absent.pt", spec=spec, source=source + b"changed"
        )


def test_inference_loader_checks_optimizer_integrity_and_restores_caller_rng(tmp_path):
    source, spec = modern_spec()
    path = tmp_path / "state.pt"
    run_training_experiment(source, spec, checkpoint_path=path)
    payload = torch.load(path, weights_only=True)
    next(iter(payload["optimizer_state"]["state"].values()))["step"].zero_()
    torch.save(payload, path)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="AdamW"):
        load_training_model_checkpoint(path, spec=spec, source=source)
    assert torch.equal(torch.get_rng_state(), rng)
