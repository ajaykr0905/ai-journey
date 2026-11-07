"""Fault-tolerant operator for bounded Project 1 training runs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from .gpt2_training import GPT2StepMetric, load_training_spec
from .gpt2_training_cli import load_source_bytes, run_and_publish
from .gpt2_training_evidence_io import load_training_evidence
from .project1_bundle import (
    build_artifact_manifest,
    seal_bundle_index,
    verify_artifact_manifest,
    verify_bundle_index,
)
from .project1_contract import (
    Project1RunError,
    contract_from_mapping,
)
from .project1_identity import attempt_id, run_id
from .project1_journal import Project1Event
from .project1_journal_io import append_event, load_event_chain
from .project1_metrics import step_metric_event
from .project1_preflight import admit_training_run, seal_preflight, verify_preflight
from .project1_state import reduce_run_state


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


def execute_fresh_run(
    *,
    source_path: Path,
    spec_path: Path,
    run_dir: Path,
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Execute one admitted fresh schedule and journal every measured update."""

    contract = _load_contract(run_dir / "contract.json")
    spec = load_training_spec(spec_path)
    source = load_source_bytes(source_path, max_bytes=contract.max_source_bytes)
    admitted = admit_training_run(contract, spec, source)
    preflight_payload = json.loads((run_dir / "preflight.json").read_bytes())
    if verify_preflight(preflight_payload, contract) != admitted:
        raise Project1RunError("stored preflight differs from current admitted inputs")
    events = load_event_chain(run_dir / "journal.json")
    state = reduce_run_state(events)
    if state.status != "running" or state.completed_step != 0 or state.event_count != 1:
        raise Project1RunError(
            "fresh execution requires an initialized untouched journal"
        )
    report = run_and_publish(
        source_path=source_path,
        spec_path=spec_path,
        checkpoint_path=run_dir / "checkpoint.pt",
        report_path=run_dir / "training-report.json",
        max_source_bytes=contract.max_source_bytes,
        max_parameters=contract.max_parameters,
    )
    sequence = 1
    for payload in report["metrics"]:
        metric = GPT2StepMetric(**payload)
        append_event(
            run_dir / "journal.json",
            step_metric_event(
                sequence=sequence,
                attempt_id=state.attempt_id,
                recorded_at_utc=recorded_at_utc,
                metric=metric,
            ),
        )
        sequence += 1
    checkpoint_digest = sha256((run_dir / "checkpoint.pt").read_bytes()).hexdigest()
    append_event(
        run_dir / "journal.json",
        Project1Event(
            sequence=sequence,
            attempt_id=state.attempt_id,
            kind="checkpoint",
            recorded_at_utc=recorded_at_utc,
            completed_step=report["progress"]["completed_step"],
            payload={"checkpoint_sha256": checkpoint_digest},
        ),
    )
    append_event(
        run_dir / "journal.json",
        Project1Event(
            sequence=sequence + 1,
            attempt_id=state.attempt_id,
            kind="completed",
            recorded_at_utc=recorded_at_utc,
            completed_step=report["progress"]["completed_step"],
            payload={"evidence_sha256": report["evidence_sha256"]},
        ),
    )
    return report


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def finalize_run_bundle(run_dir: Path) -> dict[str, Any]:
    """Seal a completed attempt's contract, journal, checkpoint, and report."""

    contract = _load_contract(run_dir / "contract.json")
    preflight = json.loads((run_dir / "preflight.json").read_bytes())
    verify_preflight(preflight, contract)
    state = reduce_run_state(load_event_chain(run_dir / "journal.json"))
    if state.status != "completed":
        raise Project1RunError("only a completed run can be finalized")
    load_training_evidence(run_dir / "training-report.json")
    artifacts = build_artifact_manifest(
        run_dir,
        [
            "checkpoint.pt",
            "contract.json",
            "journal.json",
            "preflight.json",
            "training-report.json",
        ],
    )
    index = seal_bundle_index(
        run_id=run_id(contract),
        contract_sha256=contract.fingerprint(),
        preflight_sha256=preflight["preflight_sha256"],
        journal_sha256=artifacts["journal.json"]["sha256"],
        artifacts=artifacts,
    )
    _atomic_write_json(run_dir / "bundle.json", index)
    return index


def verify_run_bundle(run_dir: Path) -> dict[str, Any]:
    """Independently verify a finalized Project 1 run from stored bytes."""

    if not isinstance(run_dir, Path) or run_dir.is_symlink() or not run_dir.is_dir():
        raise Project1RunError("run_dir must be a real directory")
    bundle_path = run_dir / "bundle.json"
    if bundle_path.is_symlink() or not bundle_path.is_file():
        raise Project1RunError("bundle index must be a regular non-symlink file")
    try:
        index = verify_bundle_index(json.loads(bundle_path.read_bytes()))
    except json.JSONDecodeError as exc:
        raise Project1RunError("bundle index is not valid JSON") from exc
    verify_artifact_manifest(run_dir, index["artifacts"])
    contract = _load_contract(run_dir / "contract.json")
    if (
        index["run_id"] != run_id(contract)
        or index["contract_sha256"] != contract.fingerprint()
    ):
        raise Project1RunError("bundle identity does not match its contract")
    preflight = json.loads((run_dir / "preflight.json").read_bytes())
    verify_preflight(preflight, contract)
    if index["preflight_sha256"] != preflight["preflight_sha256"]:
        raise Project1RunError("bundle preflight identity does not match")
    state = reduce_run_state(load_event_chain(run_dir / "journal.json"))
    if state.status != "completed":
        raise Project1RunError("bundle journal is not complete")
    load_training_evidence(run_dir / "training-report.json")
    return index
