"""Pinned public-source manifests and bounded GPT-2 artifact reads."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from .gpt2_config import GPT2Config, config_from_hugging_face, require_variant


class GPT2SourceError(ValueError):
    """Raised when a GPT-2 source manifest or artifact is unsafe."""


_SHA256 = re.compile(r"[a-f0-9]{64}")
_REVISION = re.compile(r"[a-f0-9]{40}")
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class GPT2Source:
    """Immutable identities for one public GPT-2 configuration and checkpoint."""

    model_id: str
    revision: str
    config_filename: str
    config_sha256: str
    weights_filename: str
    weights_sha256: str
    weights_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.model_id
        ):
            raise GPT2SourceError("model_id must be an owner/name identifier")
        if not isinstance(self.revision, str) or not _REVISION.fullmatch(self.revision):
            raise GPT2SourceError("revision must be a 40-character Git commit")
        for name in ("config_filename", "weights_filename"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SAFE_FILENAME.fullmatch(value):
                raise GPT2SourceError(f"{name} must be a basename")
        for name in ("config_sha256", "weights_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise GPT2SourceError(f"{name} must be a SHA-256 hex digest")
        if isinstance(self.weights_bytes, bool) or not isinstance(
            self.weights_bytes, int
        ):
            raise TypeError("weights_bytes must be an integer")
        if self.weights_bytes <= 0 or self.weights_bytes > 1_000_000_000:
            raise GPT2SourceError("weights_bytes must be in (0, 1000000000]")

    def artifact_url(self, filename: str) -> str:
        """Return the immutable Hugging Face URL for a declared artifact."""

        if filename not in {self.config_filename, self.weights_filename}:
            raise GPT2SourceError("filename is not declared by this source")
        return (
            f"https://huggingface.co/{self.model_id}/resolve/{self.revision}/{filename}"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GPT2SourceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_source_manifest(path: str | Path, *, max_bytes: int = 16_384) -> GPT2Source:
    """Load one bounded strict-JSON source manifest."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if max_bytes <= 0:
        raise GPT2SourceError("max_bytes must be positive")
    source_path = Path(path)
    if source_path.is_symlink() or not source_path.is_file():
        raise GPT2SourceError("source manifest must be a regular non-symlink file")
    raw = source_path.read_bytes()
    if len(raw) > max_bytes:
        raise GPT2SourceError("source manifest exceeds the byte limit")
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GPT2SourceError("source manifest must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise GPT2SourceError("source manifest must contain a JSON object")
    expected = {field for field in GPT2Source.__dataclass_fields__}
    if set(payload) != expected:
        raise GPT2SourceError(
            f"source manifest fields must be exactly {sorted(expected)}"
        )
    return GPT2Source(**payload)


def _sha256_file(path: Path, *, expected_bytes: int, chunk_bytes: int = 1 << 20) -> str:
    if path.is_symlink() or not path.is_file():
        raise GPT2SourceError(
            f"artifact must be a regular non-symlink file: {path.name}"
        )
    if path.stat().st_size != expected_bytes:
        raise GPT2SourceError(
            f"artifact size mismatch for {path.name}: "
            f"expected {expected_bytes}, found {path.stat().st_size}"
        )
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_artifacts(
    source: GPT2Source, directory: str | Path, *, variant: str = "gpt2"
) -> GPT2Config:
    """Verify pinned artifact sizes and hashes, then return the exact config."""

    if not isinstance(source, GPT2Source):
        raise TypeError("source must be GPT2Source")
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise GPT2SourceError("artifact directory must be a non-symlink directory")
    config_path = root / source.config_filename
    config_bytes = config_path.stat().st_size if config_path.exists() else -1
    if config_bytes <= 0 or config_bytes > 65_536:
        raise GPT2SourceError("config artifact must be in (0, 65536] bytes")
    config_digest = _sha256_file(config_path, expected_bytes=config_bytes)
    if config_digest != source.config_sha256:
        raise GPT2SourceError("config artifact SHA-256 mismatch")
    try:
        config_payload = json.loads(
            config_path.read_bytes(), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GPT2SourceError("config artifact must be strict UTF-8 JSON") from exc
    config = config_from_hugging_face(config_payload)
    require_variant(config, variant)

    weights_path = root / source.weights_filename
    weights_digest = _sha256_file(weights_path, expected_bytes=source.weights_bytes)
    if weights_digest != source.weights_sha256:
        raise GPT2SourceError("weights artifact SHA-256 mismatch")
    return config
