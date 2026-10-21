from dataclasses import asdict
from hashlib import sha256

import pytest

from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_training import (
    GPT2TrainingConfig,
    GPT2TrainingError,
    GPT2TrainingSource,
    GPT2TrainingSpec,
    run_training_experiment,
)
from ai_journey.gpt2_training_evidence import (
    GPT2_TRAINING_LIMITATIONS,
    build_training_evidence,
    verify_training_evidence,
)


def run_fixture(tmp_path):
    source = b"public training text\n" * 8
    spec = GPT2TrainingSpec(
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
            optimizer_steps=2,
            micro_batch_size=2,
            accumulation_steps=2,
            sequence_length=8,
            max_learning_rate=3e-3,
            min_learning_rate=3e-4,
            warmup_steps=0,
            weight_decay=0.1,
            gradient_clip=1.0,
            seed=47,
        ),
        source=GPT2TrainingSource(
            name="Synthetic public names",
            url="https://github.com/ajaykr0905/ai-journey",
            revision="fixture-v1",
            license="CC0-1.0",
            sha256=sha256(source).hexdigest(),
        ),
    )
    return run_training_experiment(
        source, spec, checkpoint_path=tmp_path / "checkpoint.pt"
    )


def test_training_evidence_records_complete_measured_contract(tmp_path):
    run = run_fixture(tmp_path)

    evidence = build_training_evidence(run)

    assert verify_training_evidence(evidence) == evidence
    assert evidence["spec_sha256"] == run.spec.fingerprint()
    assert evidence["progress"] == {"started_step": 0, "completed_step": 2}
    assert len(evidence["metrics"]) == 2
    assert evidence["parameter_count"] == run.parameter_count
    assert evidence["runtime"]["torch_threads"] == 1
    assert evidence["limitations"] == list(GPT2_TRAINING_LIMITATIONS)


def test_training_evidence_detects_metric_tampering(tmp_path):
    evidence = build_training_evidence(run_fixture(tmp_path))
    evidence["metrics"][0]["mean_loss"] += 1

    with pytest.raises(GPT2TrainingError, match="digest"):
        verify_training_evidence(evidence)


def test_training_evidence_rejects_resealed_accounting_tampering(tmp_path):
    evidence = build_training_evidence(run_fixture(tmp_path))
    evidence["optimizer_groups"]["decay_parameters"] += 1
    unsigned = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    evidence["evidence_sha256"] = sha256(
        __import__("json")
        .dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode()
    ).hexdigest()

    with pytest.raises(GPT2TrainingError, match="parameter count"):
        verify_training_evidence(evidence)


def test_training_evidence_preserves_declared_spec(tmp_path):
    run = run_fixture(tmp_path)
    evidence = build_training_evidence(run)
    assert evidence["spec"]["model"] == asdict(run.spec.model)
    assert evidence["spec"]["training"] == asdict(run.spec.training)
    assert evidence["spec"]["source"] == asdict(run.spec.source)
