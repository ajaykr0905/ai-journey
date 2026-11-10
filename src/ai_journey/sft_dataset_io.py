"""Atomic durable publication of SFT dataset artifacts and their manifest."""

import os
from pathlib import Path
import shutil
import tempfile

from .sft_dataset_build import SFTDatasetBuild, SFTDatasetPolicy
from .sft_dataset_identity import canonical_json_bytes
from .sft_dataset_record import SFTDatasetError
from .sft_dataset_report import build_sft_manifest
from .sft_dataset_source import SFTSourceSpec


def _write_durable(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_sft_dataset(
    output: Path,
    source: SFTSourceSpec,
    policy: SFTDatasetPolicy,
    build: SFTDatasetBuild,
) -> dict:
    """Publish all three files together, or leave no destination."""
    if not isinstance(output, Path):
        raise TypeError("output must be pathlib.Path")
    if output.exists() or output.is_symlink():
        raise SFTDatasetError("SFT output already exists")
    manifest = build_sft_manifest(source, policy, build)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    published = False
    try:
        _write_durable(temporary / "dataset.jsonl", build.artifacts.dataset_jsonl)
        _write_durable(temporary / "chatml.jsonl", build.artifacts.chatml_jsonl)
        _write_durable(temporary / "manifest.json", canonical_json_bytes(manifest))
        _fsync_directory(temporary)
        os.replace(temporary, output)
        published = True
        _fsync_directory(output.parent)
        return manifest
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)
