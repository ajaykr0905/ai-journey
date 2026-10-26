"""Rank-zero checkpoint publication for replicated CPU GPT-2 workers."""

from dataclasses import asdict, dataclass
import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import zipfile
import pickle

import torch
import torch.distributed as dist

from .ddp_contracts import integer
from .checkpoint_archive import load_checkpoint_archive
from .checkpoint_journal import CheckpointJournal
from .ddp_runtime import (
    require_all_ranks,
    require_model_consensus,
    require_optimizer_consensus,
)
from .gpt2_training import gpt2_model_fingerprint
from .training_models import model_config_mapping
from .training_optimizer_state import (
    validate_adamw_state,
    validate_serialized_adamw_state,
)


def bounded_checkpoint_load(path: Path, max_bytes: int, *, expected_sha256=None):
    """Bound both archive bytes and expanded tensor storage before deserialization."""
    integer(max_bytes, "max_bytes", 1)
    return load_checkpoint_archive(
        path, max_bytes=max_bytes, expected_sha256=expected_sha256
    )


class CheckpointRejected(RuntimeError):
    """Every rank reached a common, transactional checkpoint rejection."""


def _require_checkpoint_phase(success, *, phase):
    try:
        require_all_ranks(success, phase=phase)
    except RuntimeError as error:
        if str(error) == f"at least one rank failed {phase}":
            raise CheckpointRejected(str(error)) from error
        # A broken collective is not evidence that another candidate is safe.
        raise


def optimizer_run_contract(model, optimizer, *, serialized_groups=None):
    """Bind actual AdamW computation and ordered names, not dynamic LR/state.

    These are AdamW's invariant execution/math controls. Scheduler metadata and
    the current learning rate are deliberately excluded; a checkpoint restores
    its scheduled LR while the run's schedule remains in training_config.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("journal optimizer contract requires AdamW")
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = optimizer.param_groups if serialized_groups is None else serialized_groups
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise ValueError("journal optimizer groups do not match")
    keys = (
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
    )
    declared = []
    for destination, submitted in zip(optimizer.param_groups, groups, strict=True):
        if not isinstance(submitted, dict):
            raise ValueError("journal optimizer controls must be mappings")
        try:
            ordered_names = [
                names[id(parameter)] for parameter in destination["params"]
            ]
        except KeyError as error:
            raise ValueError(
                "journal optimizer binds an unknown model parameter"
            ) from error
        declared.append(
            {
                "parameter_names": ordered_names,
                "controls": {key: submitted[key] for key in keys if key in submitted},
            }
        )
    bound = [name for group in declared for name in group["parameter_names"]]
    if len(bound) != len(set(bound)) or set(bound) != set(names.values()):
        raise ValueError("journal optimizer must bind every named parameter once")
    result = {
        "optimizer_type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "groups": declared,
    }
    # Validate finite JSON and normalize tuple-valued controls consistently.
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))


def checkpoint_run_sha256(model, config, source: bytes, *, optimizer) -> str:
    """Bind journal ownership to the complete typed replicated run contract."""
    contract = {
        "schema": "ai-journey-ddp-journal-run-v2",
        "world_size": dist.get_world_size(),
        "model_config": model_config_mapping(model.config),
        "model_dtypes": {
            name: str(value.dtype) for name, value in model.state_dict().items()
        },
        "training_config": asdict(config),
        "source_sha256": sha256(source).hexdigest(),
        "optimizer_contract": optimizer_run_contract(model, optimizer),
    }
    return sha256(
        json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_optimizer_state(model, optimizer, *, step: int):
    """Require finite model values and the shared resumable CPU AdamW contract."""
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("checkpoint model contains nonfinite values")
    validate_adamw_state(model, optimizer, step=step)


def _exact_contract(actual, expected) -> bool:
    """Do not let Python's bool/int/float aliases change a restart contract."""
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True, allow_nan=False
    )


def _validate_model_schema(state, model) -> None:
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("checkpoint model fields do not match")
    for name, value in state.items():
        target = expected[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.shape != target.shape
            or value.dtype != target.dtype
            or value.device != target.device
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("checkpoint model tensor schema does not match")


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    config,
    source: bytes,
    step: int,
    journal_directory: Path | None = None,
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
            journal = None
            if journal_directory is not None:
                if not isinstance(
                    journal_directory, Path
                ) or path.resolve().is_relative_to(journal_directory.resolve()):
                    raise ValueError(
                        "latest checkpoint must be outside its journal directory"
                    )
                journal = CheckpointJournal(
                    journal_directory,
                    run_sha256=checkpoint_run_sha256(
                        model, config, source, optimizer=optimizer
                    ),
                )
                recorded = journal.recorded_step
                if recorded is not None and step <= recorded:
                    raise ValueError(
                        "journal checkpoint publication must advance recorded progress"
                    )
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
            if journal is not None:
                journal.publish(temporary, step=step)
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
    expected_sha256=None,
    expected_step=None,
    expected_optimizer_contract=None,
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
        payload = bounded_checkpoint_load(
            path, max_bytes, expected_sha256=expected_sha256
        )
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
        if expected_step is not None:
            integer(expected_step, "expected_step")
            if payload["step"] != expected_step:
                raise ValueError("checkpoint progress differs from journal entry")
        integer(payload["world_size"], "world_size", 1)
        if (
            payload["step"] > config.optimizer_steps
            or payload["world_size"] != dist.get_world_size()
            or not _exact_contract(
                payload["model_config"], model_config_mapping(model.config)
            )
            or not _exact_contract(payload["training_config"], asdict(config))
            or payload["source_sha256"] != sha256(source).hexdigest()
        ):
            raise ValueError("checkpoint run contract does not match")
        if (
            not isinstance(payload["rank_rngs"], list)
            or len(payload["rank_rngs"]) != dist.get_world_size()
        ):
            raise ValueError("rank RNG states do not match world size")
        _validate_model_schema(payload["model"], model)
        if expected_optimizer_contract is not None and not _exact_contract(
            optimizer_run_contract(
                model, optimizer, serialized_groups=payload["optimizer"]["param_groups"]
            ),
            expected_optimizer_contract,
        ):
            raise ValueError(
                "checkpoint optimizer invariants differ from declared journal run"
            )
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
    _require_checkpoint_phase(valid, phase="checkpoint validation")
    saved = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        torch.get_rng_state().clone(),
    )
    restored = True
    try:
        validate_serialized_adamw_state(
            model, optimizer, payload["optimizer"], step=payload["step"]
        )
        model.load_state_dict(payload["model"], strict=True)
        if gpt2_model_fingerprint(model) != payload["model_sha256"]:
            raise ValueError("checkpoint model fingerprint does not match")
        optimizer.load_state_dict(payload["optimizer"])
        validate_optimizer_state(model, optimizer, step=payload["step"])
        torch.set_rng_state(payload["rank_rngs"][dist.get_rank()])
    except (ValueError, RuntimeError, KeyError, TypeError):
        restored = False
    try:
        _require_checkpoint_phase(restored, phase="checkpoint restoration")
        require_model_consensus(model)
        require_optimizer_consensus(optimizer)
    except Exception:
        model.load_state_dict(saved[0])
        optimizer.load_state_dict(saved[1])
        torch.set_rng_state(saved[2])
        raise
    return payload["step"]


@dataclass(frozen=True)
class JournalRecovery:
    step: int
    recorded_step: int
    checkpoint_sha256: str


def load_checkpoint_journal(
    directory: Path,
    *,
    model,
    optimizer,
    config,
    source: bytes,
    max_bytes: int = 64 * 1024 * 1024,
) -> JournalRecovery:
    """Collectively restore the newest fully valid same-run byte snapshot.

    Rank zero enumerates only committed byte-intact candidates. Every rank then
    SHA-binds and validates each snapshot through the ordinary typed restore
    path. A semantic rejection rolls back before the next candidate; transport
    failures are never treated as recoverable corruption.
    """
    selection = [None]
    expected_optimizer = None
    valid = True
    try:
        expected_optimizer = optimizer_run_contract(model, optimizer)
    except (ValueError, TypeError, KeyError, OverflowError):
        valid = False
    require_all_ranks(valid, phase="journal optimizer contract")
    if dist.get_rank() == 0:
        try:
            if (
                not isinstance(directory, Path)
                or not (directory / "manifest.json").is_file()
            ):
                raise ValueError("journal resume requires an existing manifest")
            journal = CheckpointJournal(
                directory,
                run_sha256=checkpoint_run_sha256(
                    model, config, source, optimizer=optimizer
                ),
            )
            selection[0] = (
                journal.recorded_step,
                [
                    (item.step, str(item.path), item.sha256)
                    for item in journal.candidates()
                ],
            )
            valid = bool(selection[0][1])
        except (OSError, ValueError, RuntimeError, TypeError):
            valid = False
    require_all_ranks(valid, phase="checkpoint journal candidates")
    dist.broadcast_object_list(selection, src=0)
    recorded, candidates = selection[0]
    for step, path, digest in candidates:
        try:
            restored = load_checkpoint(
                Path(path),
                model=model,
                optimizer=optimizer,
                config=config,
                source=source,
                max_bytes=max_bytes,
                expected_sha256=digest,
                expected_step=step,
                expected_optimizer_contract=expected_optimizer,
            )
        except CheckpointRejected:
            continue
        return JournalRecovery(restored, recorded, digest)
    raise RuntimeError("no fully valid same-run journal checkpoint remains")
