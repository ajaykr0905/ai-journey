"""Typed rank-local RNG recovery around the existing sealed expert payload."""

import copy
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist

from .checkpoint_archive import load_checkpoint_archive, MAX_CHECKPOINT_RECORDS
from .checkpoint_journal import _read_bytes
from .ddp_runtime import (
    require_all_ranks,
    require_model_consensus,
    require_optimizer_consensus,
)
from .moe_checkpoint import (
    _contract,
    _digest,
    read_moe_checkpoint,
    restore_moe_payload,
    save_moe_checkpoint,
)
from .moe_ddp_training import validate_moe_ddp_optimizer

SCHEMA = "ai-journey-replicated-moe-checkpoint-v1"
MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class MoEDDPCheckpointIdentity:
    payload_sha256: str
    archive_sha256: str


def moe_ddp_training_contract(config, data_contract):
    return _contract(
        {
            "schema": "ai-journey-replicated-moe-run-v1",
            "config": config.to_mapping(),
            "data_contract": _contract(data_contract),
        }
    )


def _inputs(session, source, data_contract):
    if type(source) is not bytes or not 1 <= len(source) <= 1024 * 1024:
        raise ValueError("MoE DDP checkpoint source must be bounded bytes")
    if session.poisoned:
        raise ValueError("poisoned MoE DDP sessions require fresh processes")
    if (
        not dist.is_initialized()
        or dist.get_backend() != "gloo"
        or dist.get_world_size() != session.config.world_size
    ):
        raise ValueError("MoE DDP checkpoint requires its declared active Gloo world")
    return moe_ddp_training_contract(session.config, data_contract), sha256(
        source
    ).hexdigest()


def save_moe_ddp_checkpoint(
    path, *, session, source: bytes, data_contract
) -> MoEDDPCheckpointIdentity:
    """Publish a boundary and return distinct payload-seal and archive-byte SHAs."""
    valid = True
    try:
        if not isinstance(path, Path):
            raise ValueError("checkpoint path must be pathlib.Path")
        contract, source_digest = _inputs(session, source, data_contract)
        step = validate_moe_ddp_optimizer(session)
        if not 0 <= step <= session.config.optimizer_steps:
            raise ValueError("MoE DDP checkpoint progress differs")
    except (ValueError, TypeError, RuntimeError, KeyError):
        valid = False
    require_all_ranks(valid, phase="MoE DDP checkpoint admission")
    identities = [None] * session.config.world_size
    dist.all_gather_object(identities, (source_digest, _digest(contract)))
    require_all_ranks(
        len(set(identities)) == 1, phase="MoE DDP checkpoint source agreement"
    )
    require_model_consensus(session.model)
    require_optimizer_consensus(session.optimizer)
    rngs = [None] * session.config.world_size
    dist.all_gather_object(rngs, torch.get_rng_state())
    published, checkpoint_identity = True, [None]
    if dist.get_rank() == 0:
        temporary = None
        try:
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError("MoE DDP checkpoint must be a regular output")
            path.parent.mkdir(parents=True, exist_ok=True)
            # Reuse the existing public writer/reader instead of copying its
            # optimizer-binding or architecture schema into this outer format.
            with tempfile.TemporaryDirectory(
                dir=path.parent, prefix=".moe-ddp-inner-"
            ) as inner_directory:
                inner_path = Path(inner_directory) / "inner.pt"
                save_moe_checkpoint(
                    inner_path,
                    model=session.model,
                    optimizer=session.optimizer,
                    step=step,
                    training_contract=contract,
                    source_sha256=source_digest,
                )
                inner = read_moe_checkpoint(inner_path)
            payload = {
                "schema": SCHEMA,
                "world_size": session.config.world_size,
                "config": session.config.to_mapping(),
                "step": step,
                "data_cursor": {"next_update_index": step},
                "source_sha256": source_digest,
                "data_contract": contract["data_contract"],
                "inner": inner,
                "rank_rngs": rngs,
            }
            payload_digest = _digest(payload)
            payload["checkpoint_sha256"] = payload_digest
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as handle:
                temporary = Path(handle.name)
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            if temporary.stat().st_size > MAX_BYTES:
                raise ValueError("MoE DDP checkpoint exceeds archive byte budget")
            checkpoint_identity[0] = (
                payload_digest,
                sha256(_read_bytes(temporary, MAX_BYTES)).hexdigest(),
            )
            os.replace(temporary, path)
            temporary = None
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except (OSError, ValueError, RuntimeError, TypeError):
            published = False
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    require_all_ranks(published, phase="MoE DDP checkpoint publication")
    dist.broadcast_object_list(checkpoint_identity, src=0)
    return MoEDDPCheckpointIdentity(*checkpoint_identity[0])


def load_moe_ddp_checkpoint(
    path, *, session, source: bytes, data_contract, expected_archive_sha256=None
) -> int:
    """Restore a full run; expected_archive_sha256 binds file bytes, not its seal."""
    payload, valid = None, True
    try:
        contract, source_digest = _inputs(session, source, data_contract)
        validate_moe_ddp_optimizer(session)
        payload = load_checkpoint_archive(
            path,
            max_bytes=MAX_BYTES,
            max_records=MAX_CHECKPOINT_RECORDS,
            expected_sha256=expected_archive_sha256,
        )
        if type(payload) is not dict or set(payload) != {
            "schema",
            "world_size",
            "config",
            "step",
            "data_cursor",
            "source_sha256",
            "data_contract",
            "inner",
            "rank_rngs",
            "checkpoint_sha256",
        }:
            raise ValueError("MoE DDP checkpoint fields differ")
        if (
            payload["schema"] != SCHEMA
            or type(payload["world_size"]) is not int
            or payload["world_size"] != session.config.world_size
            or _digest(payload["config"]) != _digest(session.config.to_mapping())
            or payload["source_sha256"] != source_digest
            or _digest(payload["data_contract"]) != _digest(contract["data_contract"])
        ):
            raise ValueError("MoE DDP checkpoint run contract differs")
        if (
            type(payload["step"]) is not int
            or not 0 <= payload["step"] <= session.config.optimizer_steps
            or payload["inner"]["step"] != payload["step"]
        ):
            raise ValueError("MoE DDP checkpoint progress differs")
        if _digest(payload["data_cursor"]) != _digest(
            {"next_update_index": payload["step"]}
        ):
            raise ValueError("MoE DDP checkpoint data cursor differs")
        if payload["checkpoint_sha256"] != _digest(
            {
                name: value
                for name, value in payload.items()
                if name != "checkpoint_sha256"
            }
        ):
            raise ValueError("MoE DDP checkpoint seal differs")
        rngs = payload["rank_rngs"]
        if type(rngs) is not list or len(rngs) != session.config.world_size:
            raise ValueError("MoE DDP rank RNG count differs")
        for rng in rngs:
            if (
                not isinstance(rng, torch.Tensor)
                or rng.dtype != torch.uint8
                or rng.device.type != "cpu"
                or rng.shape != torch.get_rng_state().shape
            ):
                raise ValueError("MoE DDP rank RNG schema differs")
            torch.Generator(device="cpu").set_state(rng)
    except Exception:
        valid = False
    require_all_ranks(valid, phase="MoE DDP checkpoint validation")
    seals = [None] * session.config.world_size
    dist.all_gather_object(seals, payload["checkpoint_sha256"])
    require_all_ranks(
        len(set(seals)) == 1, phase="MoE DDP checkpoint snapshot agreement"
    )
    saved = (
        copy.deepcopy(session.model.state_dict()),
        copy.deepcopy(session.optimizer.state_dict()),
        torch.get_rng_state().clone(),
    )
    restored = True
    try:
        restore_moe_payload(
            payload["inner"],
            model=session.model,
            optimizer=session.optimizer,
            training_contract=contract,
            source_sha256=source_digest,
        )
        torch.set_rng_state(payload["rank_rngs"][dist.get_rank()])
        if validate_moe_ddp_optimizer(session) != payload["step"]:
            raise ValueError("MoE DDP restored optimizer differs")
    except (ValueError, RuntimeError, TypeError, KeyError):
        restored = False
    try:
        require_all_ranks(restored, phase="MoE DDP checkpoint restoration")
        require_model_consensus(session.model)
        require_optimizer_consensus(session.optimizer)
    except Exception:
        session.model.load_state_dict(saved[0], strict=True)
        session.optimizer.load_state_dict(saved[1])
        torch.set_rng_state(saved[2])
        raise
    return payload["step"]
