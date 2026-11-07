from dataclasses import replace
from hashlib import sha256

import pytest

from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_training import (
    GPT2TrainingConfig,
    GPT2TrainingSource,
    GPT2TrainingSpec,
)
from ai_journey.project1_contract import Project1RunContract
from ai_journey.project1_preflight import (
    admit_training_run,
    seal_preflight,
    verify_preflight,
)

SOURCE = b"public project one training text\n" * 4


def spec() -> GPT2TrainingSpec:
    return GPT2TrainingSpec(
        model=GPT2Config(
            vocab_size=256,
            context_length=8,
            embedding_dim=16,
            head_count=2,
            layer_count=1,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            residual_dropout=0.0,
        ),
        training=GPT2TrainingConfig(
            optimizer_steps=3,
            micro_batch_size=2,
            accumulation_steps=2,
            sequence_length=8,
            max_learning_rate=3e-3,
            min_learning_rate=3e-4,
            warmup_steps=0,
            weight_decay=0.1,
            gradient_clip=1.0,
            seed=64,
        ),
        source=GPT2TrainingSource(
            name="Public test fixture",
            url="https://github.com/ajaykr0905/ai-journey",
            revision="fixture-v1",
            license="CC0-1.0",
            sha256=sha256(SOURCE).hexdigest(),
        ),
    )


def contract() -> Project1RunContract:
    training = spec()
    return Project1RunContract(
        run_name="day-64-project1",
        spec_sha256=training.fingerprint(),
        source_sha256=training.source.sha256,
        max_source_bytes=1024,
        max_parameters=100_000,
        telemetry_interval_steps=1,
        checkpoint_interval_steps=2,
    )


def test_preflight_admits_exact_source_spec_and_resource_counts():
    result = admit_training_run(contract(), spec(), SOURCE)

    assert result.source_bytes == len(SOURCE)
    assert result.optimizer_steps == 3
    assert result.tokens_per_step == 32
    assert result.total_target_tokens == 96
    assert 0 < result.parameter_count < contract().max_parameters
    assert result.parameter_bytes == result.parameter_count * 4
    assert result.gradient_bytes == result.parameter_count * 4
    assert result.adamw_state_bytes == result.parameter_count * 8
    assert result.persistent_training_bytes == result.parameter_count * 16


@pytest.mark.parametrize(
    ("changed", "source", "message"),
    [
        ({"spec_sha256": "c" * 64}, SOURCE, "training spec"),
        ({"source_sha256": "c" * 64}, SOURCE, "digests"),
        ({"max_source_bytes": 2}, SOURCE, "byte range"),
        ({"max_parameters": 1}, SOURCE, "parameter count"),
        ({}, SOURCE + b"changed", "digests"),
    ],
)
def test_preflight_rejects_identity_and_budget_mismatches(changed, source, message):
    with pytest.raises(ValueError, match=message):
        admit_training_run(replace(contract(), **changed), spec(), source)


def test_preflight_seal_verifies_exact_accounting_and_boundary():
    admitted = admit_training_run(contract(), spec(), SOURCE)
    sealed = seal_preflight(contract(), admitted)

    assert verify_preflight(sealed, contract()) == admitted
    assert "excludes activations" in sealed["evidence_boundary"]


def test_preflight_verifier_rejects_resealed_but_false_accounting():
    admitted = admit_training_run(contract(), spec(), SOURCE)
    changed = seal_preflight(contract(), replace(admitted, parameter_bytes=1))

    with pytest.raises(ValueError, match="accounting relationships"):
        verify_preflight(changed, contract())
