"""Content-addressed artifact bundles for Project 1 training runs."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from .project1_contract import Project1RunError

_DIGEST = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[a-z0-9-]+-[0-9a-f]{16}")


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise Project1RunError("artifact path must be a non-empty string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or value != relative.as_posix():
        raise Project1RunError("artifact path must be normalized and relative")
    return relative


def build_artifact_manifest(
    root: Path,
    relative_paths: Iterable[str],
    *,
    max_file_bytes: int = 256 * 1024 * 1024,
) -> dict[str, dict[str, Any]]:
    """Hash bounded regular files without following artifact aliases."""

    if not isinstance(root, Path) or root.is_symlink() or not root.is_dir():
        raise Project1RunError("artifact root must be a real directory")
    if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int):
        raise TypeError("max_file_bytes must be an integer")
    if max_file_bytes <= 0:
        raise Project1RunError("max_file_bytes must be positive")
    names = sorted(_safe_relative_path(value).as_posix() for value in relative_paths)
    if not names or len(names) != len(set(names)):
        raise Project1RunError("artifact paths must be non-empty and unique")
    manifest = {}
    identities: set[tuple[int, int]] = set()
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise Project1RunError("artifacts must be regular non-symlink files")
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in identities:
            raise Project1RunError("artifact paths cannot be hardlink aliases")
        identities.add(identity)
        size = stat.st_size
        if size > max_file_bytes:
            raise Project1RunError("artifact exceeds max_file_bytes")
        manifest[name] = {
            "bytes": size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    return manifest


def verify_artifact_manifest(
    root: Path, manifest: Mapping[str, Mapping[str, Any]]
) -> None:
    """Re-read every declared artifact and require exact bytes and digests."""

    if not isinstance(manifest, Mapping) or not manifest:
        raise Project1RunError("artifact manifest must be a non-empty mapping")
    rebuilt = build_artifact_manifest(root, manifest.keys())
    if rebuilt != dict(manifest):
        raise Project1RunError("artifact manifest does not match stored files")
    if any(
        not isinstance(item.get("sha256"), str)
        or _DIGEST.fullmatch(item["sha256"]) is None
        for item in manifest.values()
    ):
        raise Project1RunError("artifact digest is invalid")


def seal_bundle_index(
    *,
    run_id: str,
    contract_sha256: str,
    preflight_sha256: str,
    journal_sha256: str,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal the run contract, preflight, journal, and artifacts as one unit."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise Project1RunError("run_id is invalid")
    for name, value in (
        ("contract_sha256", contract_sha256),
        ("preflight_sha256", preflight_sha256),
        ("journal_sha256", journal_sha256),
    ):
        if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
            raise Project1RunError(f"{name} must be a SHA-256 digest")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise Project1RunError("artifacts must be a non-empty manifest")
    for name, item in artifacts.items():
        _safe_relative_path(name)
        if not isinstance(item, Mapping) or set(item) != {"bytes", "sha256"}:
            raise Project1RunError("artifact manifest entry is invalid")
        if (
            isinstance(item["bytes"], bool)
            or not isinstance(item["bytes"], int)
            or item["bytes"] < 0
            or not isinstance(item["sha256"], str)
            or _DIGEST.fullmatch(item["sha256"]) is None
        ):
            raise Project1RunError("artifact manifest entry is invalid")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "contract_sha256": contract_sha256,
        "preflight_sha256": preflight_sha256,
        "journal_sha256": journal_sha256,
        "artifacts": dict(artifacts),
        "evidence_boundary": (
            "The bundle proves local artifact integrity and CPU run evidence only; "
            "it does not prove W&B upload, GPU execution, or learner activity."
        ),
    }
    payload["bundle_sha256"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def verify_bundle_index(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify exact bundle fields, artifact digests, and the bundle seal."""

    fields = {
        "schema_version",
        "run_id",
        "contract_sha256",
        "preflight_sha256",
        "journal_sha256",
        "artifacts",
        "evidence_boundary",
        "bundle_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise Project1RunError("bundle fields do not match schema")
    unsigned = {key: value for key, value in payload.items() if key != "bundle_sha256"}
    expected = sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if payload["bundle_sha256"] != expected:
        raise Project1RunError("bundle digest does not match")
    resealed = seal_bundle_index(
        run_id=payload["run_id"],
        contract_sha256=payload["contract_sha256"],
        preflight_sha256=payload["preflight_sha256"],
        journal_sha256=payload["journal_sha256"],
        artifacts=payload["artifacts"],
    )
    if resealed != dict(payload):
        raise Project1RunError("bundle contents are not canonical")
    return dict(payload)
