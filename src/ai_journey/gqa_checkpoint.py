"""Architecture-bound GQA restart and frozen inference use shared safe archives."""

from .gqa_config import GQAConfig
from .gqa_model import GroupedQueryModel
from .moe_checkpoint import load_moe_checkpoint, save_moe_checkpoint
from .moe_inference import load_moe_model_checkpoint


def save_gqa_checkpoint(
    path, *, model, optimizer, step, training_contract, source_sha256
):
    """Atomically save GQA restart state through the shared MoE archive.

    Only GroupedQueryModel is accepted (TypeError otherwise); the seal detects
    corruption, not authorship.
    """
    if not isinstance(model, GroupedQueryModel):
        raise TypeError("model must be GroupedQueryModel")
    return save_moe_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=step,
        training_contract=training_contract,
        source_sha256=source_sha256,
    )


def load_gqa_checkpoint(
    path,
    *,
    model,
    optimizer,
    training_contract,
    source_sha256,
    max_checkpoint_bytes=64 * 1024 * 1024,
):
    """Restore GQA model and optimizer state for exact restart; return the step.

    The shared MoE restorer checks provenance, architecture, optimizer controls
    and tensor schema before restoring.
    """
    if not isinstance(model, GroupedQueryModel):
        raise TypeError("model must be GroupedQueryModel")
    return load_moe_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_contract=training_contract,
        source_sha256=source_sha256,
        max_checkpoint_bytes=max_checkpoint_bytes,
    )


def load_gqa_model_checkpoint(
    path,
    *,
    config,
    training_contract,
    source,
    max_checkpoint_bytes=64 * 1024 * 1024,
    max_parameters=5_000_000,
):
    """Load a frozen eval-mode GQA model bound to its source and contract.

    A self-sealed checkpoint proves internal integrity, not authorship.
    """
    if not isinstance(config, GQAConfig):
        raise TypeError("config must be GQAConfig")
    return load_moe_model_checkpoint(
        path,
        config=config,
        training_contract=training_contract,
        source=source,
        max_checkpoint_bytes=max_checkpoint_bytes,
        max_parameters=max_parameters,
    )
