"""Atomic durable publication of HTML extraction evidence artifacts."""

import os
from pathlib import Path
import shutil
import tempfile

from .bpe import TokenizerError
from .web_corpus import ExtractionPolicy, WebCorpusResult
from .web_extraction_report import (
    build_extraction_report,
    canonical_json_bytes,
    corpus_payload,
)


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


def publish_extraction_artifacts(
    output: Path,
    result: WebCorpusResult,
    policy: ExtractionPolicy,
    *,
    snapshot_sha256: str,
) -> dict:
    """Publish corpus and manifest together or leave no partial output."""
    if not isinstance(output, Path):
        raise TypeError("HTML extraction output must be pathlib.Path")
    if output.exists():
        raise TokenizerError("HTML extraction output already exists")
    report = build_extraction_report(result, policy, snapshot_sha256=snapshot_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    published = False
    try:
        _write_durable(temporary / "corpus.jsonl", corpus_payload(result))
        _write_durable(temporary / "manifest.json", canonical_json_bytes(report))
        _fsync_directory(temporary)
        os.replace(temporary, output)
        published = True
        _fsync_directory(output.parent)
        return report
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)
