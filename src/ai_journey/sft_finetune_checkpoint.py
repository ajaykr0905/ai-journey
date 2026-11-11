"""Atomic bounded checkpoints for public-data SFT experiments."""

from __future__ import annotations

import copy
import io
import os
import re
import tempfile
from pathlib import Path

import torch

from .sft_finetune_contract import SFTFineTuneConfig, SFTFineTuneError
from .transformer_lab import DecoderLanguageModel, model_fingerprint

_DIGEST = re.compile(r"[0-9a-f]{64}")


def write_finetune_checkpoint(
    path: Path,
    model: DecoderLanguageModel,
    optimizer: torch.optim.AdamW,
    config: SFTFineTuneConfig,
    *,
    completed_steps: int,
    dataset_sha256: str,
    schedule_sha256: str,
) -> None:
    if any(
        _DIGEST.fullmatch(value) is None for value in (dataset_sha256, schedule_sha256)
    ):
        raise SFTFineTuneError("checkpoint provenance requires SHA-256 digests")
    if type(completed_steps) is not int or not 0 <= completed_steps <= config.steps:
        raise SFTFineTuneError("completed_steps is outside the configured run")
    payload = {
        "schema_version": 1,
        "config": config.as_mapping(),
        "completed_steps": completed_steps,
        "dataset_sha256": dataset_sha256,
        "schedule_sha256": schedule_sha256,
        "model_sha256": model_fingerprint(model),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.random.get_rng_state(),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    encoded = buffer.getvalue()
    if len(encoded) > 64 * 1024 * 1024:
        raise SFTFineTuneError("checkpoint exceeds the 64 MiB storage bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SFTFineTuneError("checkpoint path must be a regular non-symlink file")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_finetune_checkpoint(
    path: Path,
    model: DecoderLanguageModel,
    optimizer: torch.optim.AdamW,
    config: SFTFineTuneConfig,
) -> dict:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 64 * 1024 * 1024
    ):
        raise SFTFineTuneError("checkpoint must be a bounded regular file")
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    rng_before = torch.random.get_rng_state()
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise SFTFineTuneError("checkpoint schema is invalid")
        if value.get("config") != config.as_mapping():
            raise SFTFineTuneError("checkpoint config does not match the run")
        model.load_state_dict(value["model_state"], strict=True)
        optimizer.load_state_dict(value["optimizer_state"])
        torch.random.set_rng_state(value["torch_rng_state"])
        if model_fingerprint(model) != value["model_sha256"]:
            raise SFTFineTuneError("checkpoint model fingerprint is inconsistent")
        return {
            key: value[key]
            for key in (
                "completed_steps",
                "dataset_sha256",
                "schedule_sha256",
                "model_sha256",
            )
        }
    except Exception:
        model.load_state_dict(model_before)
        optimizer.load_state_dict(optimizer_before)
        torch.random.set_rng_state(rng_before)
        raise
