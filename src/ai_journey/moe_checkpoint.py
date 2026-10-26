"""Bounded, sealed and transactional restart state for the CPU expert lab."""

from collections.abc import Mapping
import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import torch

from .checkpoint_archive import MAX_CHECKPOINT_RECORDS, load_checkpoint_archive
from .moe_model import MoEDecoderModel
from .moe_resources import expected_checkpoint_records
from .moe_training import validate_moe_optimizer
from .training_models import training_model_fingerprint

_FIELDS = {
    "schema_version",
    "step",
    "model_config",
    "model_dtype",
    "training_contract",
    "source_sha256",
    "model_fingerprint",
    "model_state",
    "optimizer_contract",
    "optimizer_state",
    "torch_rng_state",
    "checkpoint_sha256",
}
MAX_CHECKPOINT_LOGICAL_BYTES = 128 * 1024 * 1024
MAX_CHECKPOINT_HASH_ITEMS = 100_000
MAX_CHECKPOINT_HASH_DEPTH = 64


def _contract(value) -> dict:
    if (
        not isinstance(value, Mapping)
        or not value
        or any(type(key) is not str for key in value)
    ):
        raise ValueError("training_contract must be a nonempty string-keyed mapping")
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("training_contract must be finite JSON data") from exc
    if len(encoded.encode()) > 65536:
        raise ValueError("training_contract exceeds its byte budget")
    return json.loads(encoded)


def _admit_digest_work(value) -> None:
    """Bound logical tensor work, including repeated aliases, before copying it.

    ZIP storage limits are insufficient: a four-byte storage can describe a
    billion-element expanded tensor. Charge each submitted tensor occurrence
    because canonical sealing processes each occurrence independently.
    """
    items, logical_bytes = 0, 0

    def visit(item, depth):
        nonlocal items, logical_bytes
        items += 1
        if depth > MAX_CHECKPOINT_HASH_DEPTH:
            raise ValueError("checkpoint nesting exceeds its hash-work budget")
        if items > MAX_CHECKPOINT_HASH_ITEMS:
            raise ValueError("checkpoint items exceed their hash-work budget")
        if isinstance(item, torch.Tensor):
            logical_bytes += item.numel() * item.element_size()
            if logical_bytes > MAX_CHECKPOINT_LOGICAL_BYTES:
                raise ValueError("checkpoint logical tensor work exceeds its budget")
        elif isinstance(item, dict):
            for key, part in item.items():
                visit(key, depth + 1)
                visit(part, depth + 1)
        elif isinstance(item, (tuple, list)):
            for part in item:
                visit(part, depth + 1)

    visit(value, 0)


def _digest(value) -> str:
    """Seal submitted metadata types and tensor bytes, not Python equality."""
    _admit_digest_work(value)
    digest = sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu" or item.layout != torch.strided:
                raise ValueError("checkpoint tensors must be strided CPU values")
            if (item.is_floating_point() or item.is_complex()) and not bool(
                torch.isfinite(item).all()
            ):
                raise ValueError("checkpoint tensor state must be finite")
            metadata = (str(item.dtype), tuple(item.shape))
            digest.update(b"tensor" + json.dumps(metadata).encode() + b"\0")
            digest.update(
                item.detach()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
        elif isinstance(item, dict):
            digest.update(b"dict[")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                update(key)
                update(item[key])
            digest.update(b"]")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for part in item:
                update(part)
            digest.update(b"]")
        elif item is None or type(item) in (str, int, float, bool):
            digest.update(
                type(item).__name__.encode()
                + b":"
                + json.dumps(item, allow_nan=False).encode()
                + b"\0"
            )
        else:
            raise ValueError("checkpoint contains unsupported state types")

    update(value)
    return digest.hexdigest()


def _source_digest(value):
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")


def _validate_payload(payload) -> None:
    """Apply identical metadata and seal admission to every restore entry point."""
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("MoE checkpoint fields do not match")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("MoE checkpoint schema is unsupported")
    if type(payload["step"]) is not int or not 0 <= payload["step"] <= 1_000_000:
        raise ValueError("MoE checkpoint step is invalid")
    _source_digest(payload["source_sha256"])
    if payload["checkpoint_sha256"] != _digest(
        {name: value for name, value in payload.items() if name != "checkpoint_sha256"}
    ):
        raise ValueError("MoE checkpoint byte seal does not match")


def optimizer_contract(model, optimizer) -> list[dict]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        {
            "names": [names[id(parameter)] for parameter in group["params"]],
            "controls": {
                name: value for name, value in group.items() if name != "params"
            },
        }
        for group in optimizer.param_groups
    ]


def save_moe_checkpoint(
    path: Path,
    *,
    model: MoEDecoderModel,
    optimizer,
    step: int,
    training_contract: Mapping,
    source_sha256: str,
) -> None:
    """Atomically persist the complete declared model/optimizer/CPU-RNG boundary.

    The caller owns data scheduling and supplies its full immutable training
    contract. This seal detects corruption, not authorship or authenticity.
    """
    if not isinstance(path, Path) or not isinstance(model, MoEDecoderModel):
        raise TypeError("checkpoint requires a Path and MoEDecoderModel")
    if expected_checkpoint_records(model.config) > MAX_CHECKPOINT_RECORDS:
        raise ValueError("MoE checkpoint exceeds the declared record-work budget")
    if (
        type(step) is not int
        or not 0 <= step <= 1_000_000
        or validate_moe_optimizer(model, optimizer) != step
    ):
        raise ValueError("checkpoint step does not match global optimizer progress")
    _source_digest(source_sha256)
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if len(dtypes) != 1 or next(iter(dtypes)) not in (torch.float32, torch.float64):
        raise ValueError("checkpoint model requires a single CPU floating dtype")
    payload = {
        "schema_version": 1,
        "step": step,
        "model_config": model.config.to_mapping(),
        "model_dtype": str(next(iter(dtypes))),
        "training_contract": _contract(training_contract),
        "source_sha256": source_sha256,
        "model_fingerprint": training_model_fingerprint(model),
        "model_state": model.state_dict(),
        "optimizer_contract": optimizer_contract(model, optimizer),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    payload["checkpoint_sha256"] = _digest(payload)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("checkpoint output must be a regular non-symlink path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_moe_checkpoint(
    path: Path, *, max_checkpoint_bytes: int = 64 * 1024 * 1024
) -> dict:
    """Read a bounded weights-only archive and verify exact submitted bytes."""
    if not isinstance(path, Path):
        raise TypeError("checkpoint path must be pathlib.Path")
    if type(max_checkpoint_bytes) is not int or max_checkpoint_bytes <= 0:
        raise ValueError("max_checkpoint_bytes must be positive")
    try:
        payload = load_checkpoint_archive(
            path, max_bytes=max_checkpoint_bytes, max_records=MAX_CHECKPOINT_RECORDS
        )
    except Exception as exc:
        raise ValueError(
            "MoE checkpoint is not a bounded weights-only archive"
        ) from exc
    _validate_payload(payload)
    return payload


def restore_moe_payload(
    payload: dict,
    *,
    model: MoEDecoderModel,
    optimizer,
    training_contract: Mapping,
    source_sha256: str,
) -> int:
    """Restore only after provenance, exact schema and optimizer-binding checks."""
    if not isinstance(model, MoEDecoderModel):
        raise TypeError("model must be MoEDecoderModel")
    _validate_payload(payload)
    _source_digest(source_sha256)
    if (
        payload["source_sha256"] != source_sha256
        or _contract(payload["training_contract"]) != _contract(training_contract)
        or _digest(payload["training_contract"])
        != _digest(_contract(training_contract))
    ):
        raise ValueError("MoE checkpoint source/training contract does not match")
    if _digest(payload["model_config"]) != _digest(model.config.to_mapping()):
        raise ValueError("MoE checkpoint architecture does not match")
    if _digest(payload["optimizer_contract"]) != _digest(
        optimizer_contract(model, optimizer)
    ):
        raise ValueError("MoE checkpoint optimizer controls do not match")
    expected = model.state_dict()
    state = payload["model_state"]
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("MoE checkpoint model fields do not match")
    for name, tensor in state.items():
        target = expected[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != target.shape
            or tensor.dtype != target.dtype
            or tensor.device != target.device
            or str(tensor.dtype) != payload["model_dtype"]
        ):
            raise ValueError("MoE checkpoint tensor schema does not match")
    saved = (
        copy.deepcopy(expected),
        copy.deepcopy(optimizer.state_dict()),
        torch.get_rng_state().clone(),
    )
    try:
        model.load_state_dict(state, strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if validate_moe_optimizer(model, optimizer) != payload["step"]:
            raise ValueError("MoE checkpoint optimizer progress does not match")
        if _digest(optimizer_contract(model, optimizer)) != _digest(
            payload["optimizer_contract"]
        ):
            raise ValueError("restored MoE optimizer controls do not match")
        if training_model_fingerprint(model) != payload["model_fingerprint"]:
            raise ValueError("MoE checkpoint model fingerprint does not match")
    except Exception:
        model.load_state_dict(saved[0], strict=True)
        optimizer.load_state_dict(saved[1])
        torch.set_rng_state(saved[2])
        raise
    return payload["step"]


def load_moe_checkpoint(
    path: Path,
    *,
    model: MoEDecoderModel,
    optimizer,
    training_contract: Mapping,
    source_sha256: str,
    max_checkpoint_bytes: int = 64 * 1024 * 1024,
) -> int:
    return restore_moe_payload(
        read_moe_checkpoint(path, max_checkpoint_bytes=max_checkpoint_bytes),
        model=model,
        optimizer=optimizer,
        training_contract=training_contract,
        source_sha256=source_sha256,
    )
