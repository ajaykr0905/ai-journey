"""Independent structural and cryptographic verification of MinHash evidence."""

from hashlib import sha256
import json
import math
from pathlib import Path

from .bpe import TokenizerError
from .minhash_dedup import MinHashDedupPolicy
from .minhash_report import canonical_json_bytes


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise TokenizerError("duplicate MinHash artifact field")
        value[key] = item
    return value


def _is_digest(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_bounded(path: Path, limit: int) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise TokenizerError("MinHash artifact must be a regular file")
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
    except OSError as exc:
        raise TokenizerError("unable to read MinHash artifact") from exc
    if len(payload) > limit:
        raise TokenizerError("MinHash artifact exceeds its byte budget")
    return payload


def _load_json(payload: bytes, label: str):
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TokenizerError(f"invalid MinHash {label} JSON") from exc


def _verify_candidates(report: dict, policy: MinHashDedupPolicy) -> set[str]:
    fields = {
        "left_id",
        "right_id",
        "minhash_similarity",
        "jaccard_similarity",
        "duplicate",
    }
    seen = set()
    duplicate_ids = set()
    for row in report["candidates"]:
        if not isinstance(row, dict) or set(row) != fields:
            raise TokenizerError("MinHash candidate fields are invalid")
        pair = (row["left_id"], row["right_id"])
        similarities = (row["minhash_similarity"], row["jaccard_similarity"])
        if (
            any(not isinstance(value, str) or not value for value in pair)
            or pair[0] >= pair[1]
            or pair in seen
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
                for value in similarities
            )
            or type(row["duplicate"]) is not bool
            or row["duplicate"]
            != (row["jaccard_similarity"] >= policy.duplicate_threshold)
        ):
            raise TokenizerError("MinHash candidate values are invalid")
        seen.add(pair)
        if row["duplicate"]:
            duplicate_ids.update(pair)
    return duplicate_ids


def _verify_clusters(report: dict) -> tuple[set[str], set[str]]:
    canonical_ids = set()
    removed_ids = set()
    for row in report["clusters"]:
        if not isinstance(row, dict) or set(row) != {
            "canonical_id",
            "duplicate_ids",
        }:
            raise TokenizerError("MinHash cluster fields are invalid")
        canonical = row["canonical_id"]
        duplicates = row["duplicate_ids"]
        if (
            not isinstance(canonical, str)
            or not canonical
            or not isinstance(duplicates, list)
            or not duplicates
            or any(
                not isinstance(value, str) or value <= canonical for value in duplicates
            )
            or len(set(duplicates)) != len(duplicates)
            or canonical in canonical_ids
            or canonical in removed_ids
            or canonical in duplicates
            or removed_ids.intersection(duplicates)
        ):
            raise TokenizerError("MinHash cluster values are invalid")
        canonical_ids.add(canonical)
        removed_ids.update(duplicates)
    return canonical_ids, removed_ids


def verify_minhash_artifacts(directory: Path) -> dict:
    """Verify exact inventory, report seal, removal counts, and corpus bytes."""
    if not isinstance(directory, Path):
        raise TypeError("MinHash artifact root must be pathlib.Path")
    if not directory.is_dir() or directory.is_symlink():
        raise TokenizerError("MinHash artifact root must be a regular directory")
    entries = tuple(directory.iterdir())
    if {entry.name for entry in entries} != {
        "manifest.json",
        "deduplicated-corpus.jsonl",
    }:
        raise TokenizerError("MinHash artifact inventory is invalid")
    report = _load_json(
        _read_bounded(directory / "manifest.json", 8 * 1024 * 1024), "manifest"
    )
    fields = {
        "schema_version",
        "input",
        "policy",
        "metrics",
        "candidates",
        "clusters",
        "corpus",
        "evidence_boundary",
        "report_sha256",
    }
    if (
        not isinstance(report, dict)
        or set(report) != fields
        or type(report["schema_version"]) is not int
        or report["schema_version"] != 1
        or not _is_digest(report["report_sha256"])
        or not isinstance(report["evidence_boundary"], str)
        or not report["evidence_boundary"]
    ):
        raise TokenizerError("MinHash manifest fields are invalid")
    unsealed = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256(canonical_json_bytes(unsealed)).hexdigest() != report["report_sha256"]:
        raise TokenizerError("MinHash report seal mismatch")
    source = report["input"]
    if (
        not isinstance(source, dict)
        or set(source) != {"dataset_report_sha256", "shard_sha256s", "documents"}
        or not _is_digest(source["dataset_report_sha256"])
        or not isinstance(source["shard_sha256s"], list)
        or not source["shard_sha256s"]
        or any(not _is_digest(value) for value in source["shard_sha256s"])
        or type(source["documents"]) is not int
        or source["documents"] < 1
    ):
        raise TokenizerError("MinHash input identity is invalid")
    try:
        policy = MinHashDedupPolicy(**report["policy"])
    except (TypeError, TokenizerError) as exc:
        raise TokenizerError("MinHash manifest policy is invalid") from exc
    if set(report["policy"]) != {
        "shingle_width",
        "num_hashes",
        "seed",
        "bands",
        "duplicate_threshold",
    }:
        raise TokenizerError("MinHash manifest policy fields are invalid")
    if not isinstance(report["candidates"], list) or not isinstance(
        report["clusters"], list
    ):
        raise TokenizerError("MinHash pair or cluster evidence is invalid")
    duplicate_pair_ids = _verify_candidates(report, policy)
    canonical_ids, removed_ids = _verify_clusters(report)
    if (canonical_ids | removed_ids) - duplicate_pair_ids:
        raise TokenizerError("MinHash clusters lack accepted pair evidence")
    metrics = report["metrics"]
    metric_fields = {
        "input_documents",
        "candidate_pairs",
        "duplicate_pairs",
        "duplicate_clusters",
        "kept_documents",
        "removed_documents",
        "removed_fraction",
    }
    if (
        not isinstance(metrics, dict)
        or set(metrics) != metric_fields
        or any(
            type(metrics[name]) is not int or metrics[name] < 0
            for name in metric_fields - {"removed_fraction"}
        )
        or not isinstance(metrics["removed_fraction"], float)
        or not math.isfinite(metrics["removed_fraction"])
        or metrics["input_documents"] != source["documents"]
        or metrics["candidate_pairs"] != len(report["candidates"])
        or metrics["duplicate_pairs"]
        != sum(row["duplicate"] for row in report["candidates"])
        or metrics["duplicate_clusters"] != len(report["clusters"])
        or metrics["removed_documents"] != len(removed_ids)
        or metrics["kept_documents"] + metrics["removed_documents"]
        != metrics["input_documents"]
        or metrics["removed_fraction"]
        != metrics["removed_documents"] / metrics["input_documents"]
    ):
        raise TokenizerError("MinHash removal metrics are invalid")
    corpus = report["corpus"]
    if (
        not isinstance(corpus, dict)
        or set(corpus) != {"path", "documents", "artifact_bytes", "sha256"}
        or corpus["path"] != "deduplicated-corpus.jsonl"
        or type(corpus["documents"]) is not int
        or corpus["documents"] != metrics["kept_documents"]
        or type(corpus["artifact_bytes"]) is not int
        or not 1 <= corpus["artifact_bytes"] <= 64 * 1024 * 1024
        or not _is_digest(corpus["sha256"])
    ):
        raise TokenizerError("MinHash corpus descriptor is invalid")
    payload = _read_bounded(
        directory / "deduplicated-corpus.jsonl", corpus["artifact_bytes"]
    )
    if (
        len(payload) != corpus["artifact_bytes"]
        or sha256(payload).hexdigest() != corpus["sha256"]
        or not payload.endswith(b"\n")
    ):
        raise TokenizerError("MinHash corpus identity mismatch")
    rows = [_load_json(line, "corpus row") for line in payload.splitlines()]
    kept_ids = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "document_id",
            "source",
            "source_sha256",
            "text",
        }:
            raise TokenizerError("MinHash corpus row fields are invalid")
        identity = row["document_id"]
        if (
            not isinstance(identity, str)
            or not identity
            or identity in kept_ids
            or not isinstance(row["source"], str)
            or not row["source"]
            or not _is_digest(row["source_sha256"])
            or not isinstance(row["text"], str)
            or not row["text"]
        ):
            raise TokenizerError("MinHash corpus row values are invalid")
        kept_ids.add(identity)
    if (
        len(rows) != corpus["documents"]
        or canonical_ids - kept_ids
        or removed_ids.intersection(kept_ids)
    ):
        raise TokenizerError("MinHash corpus removal coverage mismatch")
    return report
