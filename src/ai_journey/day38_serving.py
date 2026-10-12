"""Read-only operations on verified Day 38 training artifacts."""

from dataclasses import dataclass
from math import isclose
from pathlib import Path

import torch

from .serving_benchmark import _publish
from .transformer_cache import generate_cached

from .day38_training import (
    Day38DataConfig,
    Day38TrainingConfig,
    _preserve_random_state,
    load_day38_report,
    prepare_bounded_corpus,
)
from .gpt_bigram import CorpusSource, load_verified_corpus
from .transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TokenCorpus,
    TransformerLabError,
    build_optimizer,
    expected_parameter_count,
    load_training_checkpoint,
    model_fingerprint,
    evaluate_nll,
)


@dataclass
class TrainedArtifact:
    """Weights and corpus bound to a verified training report."""

    model: DecoderLanguageModel
    corpus: TokenCorpus
    report: dict


def load_trained_artifact(
    report_path: Path,
    checkpoint_path: Path,
    corpus_path: Path,
    *,
    max_parameters: int = 1_000_000,
) -> TrainedArtifact:
    """Validate source, codec, sampler, step, and weights before returning a model."""
    report = load_day38_report(report_path)
    evidence = report["experiment"]
    settings = dict(evidence["config"])
    data = Day38DataConfig(**settings.pop("data"))
    config = Day38TrainingConfig(data=data, **settings)
    architecture = config.model_config(evidence["vocab_size"])
    if (
        isinstance(max_parameters, bool)
        or not isinstance(max_parameters, int)
        or max_parameters <= 0
        or expected_parameter_count(architecture) > max_parameters
    ):
        raise TransformerLabError("trained model exceeds the parameter budget")
    source = CorpusSource(**evidence["source"])
    text = load_verified_corpus(corpus_path, source)
    corpus = prepare_bounded_corpus(text, source, data).corpus
    if (
        corpus.fingerprint() != evidence["corpus_fingerprint"]
        or corpus.codec.fingerprint() != evidence["codec_fingerprint"]
    ):
        raise TransformerLabError("training report does not match the corpus")
    with _preserve_random_state():
        model = DecoderLanguageModel(architecture)
        training = config.training_config()
        optimizer = build_optimizer(model, training)
        cursor = BatchCursor(
            corpus.train_tokens,
            block_size=data.block_size,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        step = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            cursor=cursor,
            training_config=training,
            corpus_fingerprint=corpus.fingerprint(),
        )
        if (
            step != evidence["completed_steps"]
            or model_fingerprint(model) != evidence["model_fingerprint"]
        ):
            raise TransformerLabError("checkpoint does not match the training report")
    model.eval()
    return TrainedArtifact(model, corpus, report)


def generate_from_artifact(
    artifact: TrainedArtifact,
    prompt: str,
    *,
    new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int | None = 20,
    seed: int = 3800,
    cached: bool = False,
) -> dict:
    """Generate a bounded sample with provenance and caller RNG isolation."""
    if not isinstance(artifact, TrainedArtifact):
        raise TypeError("artifact must be TrainedArtifact")
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 1024:
        raise TransformerLabError("prompt must contain 1..1024 characters")
    if type(new_tokens) is not int or not 1 <= new_tokens <= 256:
        raise TransformerLabError("new_tokens must be in [1, 256]")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise TransformerLabError("seed must be in [0, 2**63)")
    if not isinstance(cached, bool):
        raise TypeError("cached must be boolean")
    expected = artifact.report["experiment"]["model_fingerprint"]
    if model_fingerprint(artifact.model) != expected:
        raise TransformerLabError("live model no longer matches the training artifact")
    tokens = torch.tensor([artifact.corpus.codec.encode(prompt)], dtype=torch.long)
    if cached:
        output = generate_cached(
            artifact.model,
            tokens,
            new_tokens=new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
    else:
        generator = torch.Generator().manual_seed(seed)
        output = artifact.model.generate(
            tokens,
            new_tokens=new_tokens,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )
    ids = output[0].tolist()
    return {
        "schema": "day38-generation-v1",
        "training_evidence_fingerprint": artifact.report["evidence_fingerprint"],
        "model_fingerprint": expected,
        "controls": {
            "prompt": prompt,
            "new_tokens": new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "seed": seed,
            "cached": cached,
        },
        "token_ids": ids,
        "text": artifact.corpus.codec.decode(ids),
    }


def write_serving_report(path: Path, report: dict) -> None:
    """Use the lab's durable atomic JSON publisher for serving evidence."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    _publish(path, report)


def audit_trained_artifact(artifact: TrainedArtifact) -> dict:
    """Independently replay the sample and remeasure both final loss metrics."""
    evidence = artifact.report["experiment"]
    config = evidence["config"]
    sample = generate_from_artifact(
        artifact,
        config["sample_prompt"],
        new_tokens=config["sample_tokens"],
        temperature=config["sample_temperature"],
        top_k=config["sample_top_k"],
        seed=config["sample_seed"],
    )
    train = evaluate_nll(artifact.model, artifact.corpus.train_tokens)
    validation = evaluate_nll(artifact.model, artifact.corpus.validation_tokens)
    checks = {
        "sample_replay": sample["token_ids"] == evidence["sample_token_ids"],
        "train_nll": isclose(
            train, evidence["final_train_nll"], abs_tol=1e-6, rel_tol=1e-6
        ),
        "validation_nll": isclose(
            validation, evidence["final_validation_nll"], abs_tol=1e-6, rel_tol=1e-6
        ),
    }
    if not all(checks.values()):
        raise TransformerLabError(
            "checkpoint audit failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )
    return {
        "schema": "day38-checkpoint-audit-v1",
        "training_evidence_fingerprint": artifact.report["evidence_fingerprint"],
        "model_fingerprint": evidence["model_fingerprint"],
        "checks": checks,
        "observed_train_nll": train,
        "observed_validation_nll": validation,
        "loss_tolerance": {"relative": 1e-6, "absolute": 1e-6},
        "passed": True,
    }
