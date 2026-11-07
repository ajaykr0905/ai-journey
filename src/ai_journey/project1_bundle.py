"""Content-addressed artifact bundles for Project 1 training runs."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from .project1_contract import Project1RunError

_DIGEST = re.compile(r"[0-9a-f]{64}")


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
