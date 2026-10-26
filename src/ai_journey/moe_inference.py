"""Freeze only provenance-checked, restart-valid CPU expert checkpoints."""

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

import torch

from .moe_checkpoint import _contract, _digest, read_moe_checkpoint, restore_moe_payload
from .moe_config import MoEConfig
from .moe_training import build_moe_optimizer, validate_moe_optimizer
from .training_models import build_training_model


def _optimizer_from_contract(model, contract):
    """Rebuild declared parameter groups, rejecting unknown or repeated names."""
    template = build_moe_optimizer(model, learning_rate=0.001)
    fields = set(template.param_groups[0]) - {"params"}
    parameters = dict(model.named_parameters())
    if not isinstance(contract, list) or not 1 <= len(contract) <= 16:
        raise ValueError("MoE optimizer contract must contain bounded groups")
    groups, seen = [], []
    for group in contract:
        if not isinstance(group, dict) or set(group) != {"names", "controls"}:
            raise ValueError("MoE optimizer group fields do not match")
        names, controls = group["names"], group["controls"]
        if (
            not isinstance(names, list)
            or not isinstance(controls, dict)
            or set(controls) != fields
            or any(type(name) is not str or name not in parameters for name in names)
        ):
            raise ValueError("MoE optimizer group names/controls are invalid")
        seen.extend(names)
        groups.append({**controls, "params": [parameters[name] for name in names]})
    if len(seen) != len(set(seen)) or set(seen) != set(parameters):
        raise ValueError("MoE optimizer contract must bind every parameter once")
    optimizer = torch.optim.AdamW(groups)
    validate_moe_optimizer(model, optimizer)
    return optimizer


def load_moe_model_checkpoint(
    path: Path,
    *,
    config: MoEConfig,
    training_contract: Mapping,
    source: bytes,
    max_checkpoint_bytes: int = 64 * 1024 * 1024,
    max_parameters: int = 5_000_000,
):
    """Validate archive/provenance/restart state, then return dropless frozen eval.

    Source bytes and the complete caller-declared training contract are required.
    A self-sealed checkpoint proves internal integrity, not authorship, public
    licensing or that somebody performed a manual learning exercise.
    """
    if not isinstance(config, MoEConfig):
        raise TypeError("config must be MoEConfig")
    if type(source) is not bytes or not 0 < len(source) <= 16 * 1024 * 1024:
        raise ValueError("source must be nonempty bounded bytes")
    payload = read_moe_checkpoint(path, max_checkpoint_bytes=max_checkpoint_bytes)
    source_sha256 = sha256(source).hexdigest()
    if payload["source_sha256"] != source_sha256:
        raise ValueError("MoE checkpoint source bytes do not match")
    if _digest(payload["model_config"]) != _digest(config.to_mapping()) or _digest(
        payload["training_contract"]
    ) != _digest(_contract(training_contract)):
        raise ValueError("MoE checkpoint architecture/training contract does not match")
    dtypes = {"torch.float32": torch.float32, "torch.float64": torch.float64}
    if payload["model_dtype"] not in dtypes:
        raise ValueError("MoE checkpoint model dtype is unsupported")
    with torch.random.fork_rng(devices=[]):
        model = build_training_model(config, max_parameters=max_parameters).to(
            dtype=dtypes[payload["model_dtype"]]
        )
        optimizer = _optimizer_from_contract(model, payload["optimizer_contract"])
        restore_moe_payload(
            payload,
            model=model,
            optimizer=optimizer,
            training_contract=training_contract,
            source_sha256=source_sha256,
        )
    return model.eval().requires_grad_(False)
