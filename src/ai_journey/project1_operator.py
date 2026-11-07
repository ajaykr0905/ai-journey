"""Fault-tolerant operator for bounded Project 1 training runs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .gpt2_training import load_training_spec
from .gpt2_training_cli import load_source_bytes
from .project1_contract import (
    Project1RunError,
    contract_from_mapping,
)
from .project1_identity import attempt_id, run_id
from .project1_journal import Project1Event
from .project1_journal_io import append_event
from .project1_preflight import admit_training_run, seal_preflight


def _load_contract(path: Path, *, max_bytes: int = 64 * 1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise Project1RunError("contract must be a bounded regular non-symlink file")
    try:
        payload = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise Project1RunError("contract is not valid JSON") from exc
    return contract_from_mapping(payload)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def initialize_run(
    *,
    source_path: Path,
    spec_path: Path,
    contract_path: Path,
    run_dir: Path,
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Admit inputs and atomically initialize the first attempt directory."""

    for name, path in (
        ("source_path", source_path),
        ("spec_path", spec_path),
        ("contract_path", contract_path),
        ("run_dir", run_dir),
    ):
        if not isinstance(path, Path):
            raise TypeError(f"{name} must be pathlib.Path")
    if run_dir.exists() or run_dir.is_symlink():
        raise Project1RunError("run_dir must not already exist")
    source = load_source_bytes(source_path)
    spec = load_training_spec(spec_path)
    contract = _load_contract(contract_path)
    preflight = admit_training_run(contract, spec, source)
    sealed_preflight = seal_preflight(contract, preflight)
    identity = run_id(contract)
    first_attempt = attempt_id(identity, 1)
    parent = run_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise Project1RunError("run directory parent must be a real directory")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}-", dir=parent))
    try:
        _write_json(temporary / "contract.json", contract.to_mapping())
        _write_json(temporary / "preflight.json", sealed_preflight)
        append_event(
            temporary / "journal.json",
            Project1Event(
                sequence=0,
                attempt_id=first_attempt,
                kind="started",
                recorded_at_utc=recorded_at_utc,
                completed_step=0,
                payload={
                    "run_id": identity,
                    "contract_sha256": contract.fingerprint(),
                    "preflight_sha256": sealed_preflight["preflight_sha256"],
                },
            ),
        )
        os.rename(temporary, run_dir)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return {
        "run_id": identity,
        "attempt_id": first_attempt,
        "contract_sha256": contract.fingerprint(),
        "preflight_sha256": sealed_preflight["preflight_sha256"],
        "run_dir": str(run_dir),
    }
