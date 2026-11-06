"""Shared model loading boundary for public byte-level evaluations."""

from hashlib import sha256
from pathlib import Path

import torch

from .ddp_pipeline import load_serving_model
from .gpt2_training import load_training_spec
from .hellaswag_io import read_evaluation_bytes
from .training_artifact import load_training_model_checkpoint
from .training_models import build_training_model, expected_training_parameters


def load_evaluation_model(
    source: Path,
    spec_path: Path,
    checkpoint: Path | None,
    *,
    checkpoint_format: str = "distributed",
):
    """Load one bounded model only after binding it to its training source."""
    spec = load_training_spec(spec_path)
    source_bytes = read_evaluation_bytes(
        source, maximum=16 * 1024 * 1024, name="training source"
    )
    if sha256(source_bytes).hexdigest() != spec.source.sha256:
        raise ValueError("model source does not match the training specification")
    if expected_training_parameters(spec.model) > 1_000_000:
        raise ValueError("evaluation model exceeds the CPU parameter budget")
    if checkpoint is not None:
        if checkpoint_format == "local":
            return load_training_model_checkpoint(
                checkpoint, spec=spec, source=source_bytes, max_parameters=1_000_000
            )
        if checkpoint_format != "distributed":
            raise ValueError("unsupported evaluation checkpoint format")
        return load_serving_model(checkpoint, spec, source_bytes)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.training.seed)
        return (
            build_training_model(spec.model, max_parameters=1_000_000).double().eval()
        )
