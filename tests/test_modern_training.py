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
