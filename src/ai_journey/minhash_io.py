"""Atomic durable publication of MinHash deduplication evidence."""

import os
from pathlib import Path
import shutil
import tempfile

from .bpe import TokenizerError
from .minhash_corpus import Project1Corpus
from .minhash_dedup import MinHashDedupPolicy, MinHashDedupResult
from .minhash_report import (
    build_minhash_report,
    canonical_json_bytes,
    deduplicated_corpus_payload,
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


def publish_minhash_artifacts(
    output: Path,
    corpus: Project1Corpus,
    policy: MinHashDedupPolicy,
    result: MinHashDedupResult,
) -> dict:
    """Publish the deduplicated corpus and report together or not at all."""
    if not isinstance(output, Path):
        raise TypeError("MinHash output must be pathlib.Path")
    if output.exists():
        raise TokenizerError("MinHash output already exists")
    report = build_minhash_report(corpus, policy, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    published = False
    try:
        _write_durable(
            temporary / "deduplicated-corpus.jsonl",
            deduplicated_corpus_payload(result),
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
