"""Load a provenance-checked local training checkpoint for CPU inference."""

from pathlib import Path

import torch

from .gpt2_training import (
    GPT2BatchCursor,
    GPT2ByteCorpus,
    GPT2TrainingError,
    GPT2TrainingSpec,
    build_gpt2_optimizer,
    load_training_checkpoint,
)
from .training_models import TrainingModel, build_training_model


def load_training_model_checkpoint(
    path: Path,
    *,
    spec: GPT2TrainingSpec,
    source: bytes,
    max_checkpoint_bytes: int = 64 * 1024 * 1024,
    max_parameters: int = 5_000_000,
) -> TrainingModel:
    """Validate the complete restart contract and return a frozen eval model.

    Inference consumers must supply the original public source and specification;
    a tensor state alone does not establish the model's training provenance.
    Optimizer and cursor state are validated using the same transactional loader
    as training, then discarded. The caller's random-number state is preserved.
    """
    if not isinstance(spec, GPT2TrainingSpec):
        raise TypeError("spec must be GPT2TrainingSpec")
    corpus = GPT2ByteCorpus.from_bytes(source)
    if corpus.source_sha256 != spec.source.sha256:
        raise GPT2TrainingError("source bytes do not match the training spec")
    with torch.random.fork_rng(devices=[]):
        model = build_training_model(spec.model, max_parameters=max_parameters)
        optimizer = build_gpt2_optimizer(model, spec.training)
        cursor = GPT2BatchCursor(corpus, spec.training)
        load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            cursor=cursor,
            config=spec.training,
            corpus_fingerprint=corpus.fingerprint(),
            max_checkpoint_bytes=max_checkpoint_bytes,
        )
    return model.eval().requires_grad_(False)
