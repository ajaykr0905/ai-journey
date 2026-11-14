"""Canonical sealed reports for Project 1 MinHash deduplication."""

from dataclasses import asdict
from hashlib import sha256
import json
import math

from .bpe import TokenizerError
from .minhash_corpus import Project1Corpus
from .minhash_dedup import MinHashDedupPolicy, MinHashDedupResult


def canonical_json_bytes(value) -> bytes:
    """The one encoding of corpus rows, the manifest and its seal.

    Values JSON cannot represent exactly, such as NaN, raise TokenizerError.
    """
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise TokenizerError("MinHash evidence is not canonical JSON") from exc


def deduplicated_corpus_payload(result: MinHashDedupResult) -> bytes:
    """Kept documents as canonical JSONL in id order: the bytes the manifest hashes."""
    if not isinstance(result, MinHashDedupResult):
        raise TypeError("deduplicated corpus payload requires a MinHash result")
    return b"".join(
        canonical_json_bytes(
            {
                "document_id": row.document_id,
                "source": row.source,
                "source_sha256": row.source_sha256,
                "text": row.text,
            }
        )
        for row in result.documents
    )


def build_minhash_report(
    corpus: Project1Corpus,
    policy: MinHashDedupPolicy,
    result: MinHashDedupResult,
) -> dict:
    """Seal source identities, policy, pair evidence, removals, and output bytes."""
    if not isinstance(corpus, Project1Corpus):
        raise TypeError("MinHash report requires a Project1Corpus")
    if not isinstance(policy, MinHashDedupPolicy) or not isinstance(
        result, MinHashDedupResult
    ):
        raise TypeError("MinHash report requires policy and result")
    floats = [
        result.metrics.removed_fraction,
        *(
            value
            for row in result.assessments
            for value in (row.minhash_similarity, row.jaccard_similarity)
        ),
    ]
    if any(not math.isfinite(value) for value in floats):
        raise TokenizerError("MinHash report values must be finite")
    payload = deduplicated_corpus_payload(result)
    report = {
        "schema_version": 1,
        "input": {
            "dataset_report_sha256": corpus.dataset_report_sha256,
            "shard_sha256s": list(corpus.shard_sha256s),
            "documents": len(corpus.documents),
        },
        "policy": asdict(policy),
        "metrics": asdict(result.metrics),
        "candidates": [asdict(row) for row in result.assessments],
        "clusters": [
            {
                "canonical_id": row.canonical_id,
                "duplicate_ids": list(row.duplicate_ids),
            }
            for row in result.decisions
        ],
        "corpus": {
            "path": "deduplicated-corpus.jsonl",
            "documents": len(result.documents),
            "artifact_bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        },
        "evidence_boundary": (
            "deterministic CPU MinHash/LSH over a verified Project 1 dataset; "
            "no claim of learner execution, corpus quality, or production scale"
        ),
    }
    report["report_sha256"] = sha256(canonical_json_bytes(report)).hexdigest()
    return report
