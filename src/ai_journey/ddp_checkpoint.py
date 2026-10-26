"""Rank-zero checkpoint publication for replicated CPU GPT-2 workers."""

from dataclasses import asdict
import copy
from hashlib import sha256
import os
from pathlib import Path
import tempfile
import zipfile
import pickle

import torch
import torch.distributed as dist

from .ddp_contracts import integer
from .checkpoint_archive import load_checkpoint_archive
from .ddp_runtime import (
    require_all_ranks,
    require_model_consensus,
    require_optimizer_consensus,
)
from .gpt2_training import gpt2_model_fingerprint
from .training_models import model_config_mapping
from .training_optimizer_state import validate_adamw_state


def bounded_checkpoint_load(path: Path, max_bytes: int):
    """Bound both archive bytes and expanded tensor storage before deserialization."""
    integer(max_bytes, "max_bytes", 1)
    return load_checkpoint_archive(path, max_bytes=max_bytes)


def validate_optimizer_state(model, optimizer, *, step: int):
    """Require finite model values and the shared resumable CPU AdamW contract."""
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("checkpoint model contains nonfinite values")
    validate_adamw_state(model, optimizer, step=step)


def save_checkpoint(
    path: Path, *, model, optimizer, config, source: bytes, step: int
) -> str:
    integer(step, "step")
    if step > config.optimizer_steps:
        raise ValueError("step exceeds schedule")
    valid = True
    try:
        validate_optimizer_state(model, optimizer, step=step)
    except ValueError:
        valid = False
    require_all_ranks(valid, phase="checkpoint state validation")
    fingerprint = require_model_consensus(model)
    require_optimizer_consensus(optimizer)
    rngs = [None] * dist.get_world_size()
    dist.all_gather_object(rngs, torch.get_rng_state())
    success = True
    if dist.get_rank() == 0:
        temporary = None
        try:
            if path.is_symlink():
                raise ValueError("checkpoint path must not be a symlink")
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as handle:
                temporary = Path(handle.name)
                torch.save(
                    {
                        "schema_version": 1,
                        "world_size": dist.get_world_size(),
                        "step": step,
                        "model_config": model_config_mapping(model.config),
                        "training_config": asdict(config),
                        "source_sha256": sha256(source).hexdigest(),
                        "model_sha256": fingerprint,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rank_rngs": rngs,
                    },
                    handle,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except (OSError, ValueError, RuntimeError):
            success = False
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    require_all_ranks(success, phase="checkpoint publication")
    return fingerprint


def load_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    config,
    source: bytes,
    max_bytes: int = 64 * 1024 * 1024,
) -> int:
    """Validate on every rank and restore atomically with rank-local RNG state."""
    payload = None
    valid = True
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or not 0 < path.stat().st_size <= max_bytes
        ):
            raise ValueError("checkpoint exceeds file constraints")
        payload = bounded_checkpoint_load(path, max_bytes)
        expected = {
            "schema_version",
            "world_size",
            "step",
            "model_config",
            "training_config",
            "source_sha256",
            "model_sha256",
            "model",
            "optimizer",
            "rank_rngs",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
        ):
            raise ValueError("checkpoint schema does not match")
        integer(payload["step"], "step")
        if (
            payload["step"] > config.optimizer_steps
            or payload["world_size"] != dist.get_world_size()
            or payload["model_config"] != model_config_mapping(model.config)
            or payload["training_config"] != asdict(config)
            or payload["source_sha256"] != sha256(source).hexdigest()
        ):
            raise ValueError("checkpoint run contract does not match")
        if (
            not isinstance(payload["rank_rngs"], list)
            or len(payload["rank_rngs"]) != dist.get_world_size()
        ):
            raise ValueError("rank RNG states do not match world size")
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        zipfile.BadZipFile,
        pickle.UnpicklingError,
        EOFError,
    ):
        valid = False
    require_all_ranks(valid, phase="checkpoint validation")
    saved = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        torch.get_rng_state().clone(),
    )
    restored = True
    try:
        model.load_state_dict(payload["model"], strict=True)
        if gpt2_model_fingerprint(model) != payload["model_sha256"]:
            raise ValueError("checkpoint model fingerprint does not match")
        optimizer.load_state_dict(payload["optimizer"])
        validate_optimizer_state(model, optimizer, step=payload["step"])
        torch.set_rng_state(payload["rank_rngs"][dist.get_rank()])
    except (ValueError, RuntimeError, KeyError, TypeError):
        restored = False
    try:
        require_all_ranks(restored, phase="checkpoint restoration")
        require_model_consensus(model)
        require_optimizer_consensus(optimizer)
    except Exception:
        model.load_state_dict(saved[0])
        optimizer.load_state_dict(saved[1])
        torch.set_rng_state(saved[2])
        raise
    return payload["step"]
