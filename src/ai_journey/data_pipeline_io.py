"""Atomic durable publication for complete dataset build artifacts."""

import os
from pathlib import Path
import shutil
import tempfile

from .bpe import TokenizerError
from .data_pipeline import DataPipelineResult
from .data_pipeline_report import (
    build_pipeline_report,
    canonical_json_bytes,
    shard_payload,
)
from .data_pipeline_source import SourceEvidence


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


def publish_pipeline_artifacts(
    output: Path, result: DataPipelineResult, source: SourceEvidence
) -> dict:
    """Publish shards and manifest together or leave no visible dataset."""
    if not isinstance(output, Path):
        raise TypeError("output must be pathlib.Path")
    if output.exists():
        raise TokenizerError("dataset output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    published = False
    try:
        report = build_pipeline_report(result, source)
        for name, shards in (
            ("train", result.train_shards),
            ("validation", result.validation_shards),
        ):
            for shard in shards:
                _write_durable(
                    temporary / f"{name}-{shard.index:05d}.jsonl",
                    shard_payload(shard),
                )
        _write_durable(temporary / "manifest.json", canonical_json_bytes(report))
        _fsync_directory(temporary)
        os.replace(temporary, output)
        published = True
        _fsync_directory(output.parent)
        return report
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)
