"""Independent structural and cryptographic verification of dataset artifacts."""

from hashlib import sha256
import json
from pathlib import Path

from .bpe import TokenizerError
from .data_pipeline_report import canonical_json_bytes
from .tokenizer_io import tokenizer_fingerprint, tokenizer_from_state
from .tokenizer_metrics import compression_metrics


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TokenizerError("duplicate dataset artifact field")
        result[key] = value
    return result


def _read_bounded(path: Path, limit: int) -> bytes:
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
    except OSError as exc:
        raise TokenizerError(f"unable to read dataset artifact: {path.name}") from exc
    if len(payload) > limit:
        raise TokenizerError(f"dataset artifact exceeds its byte budget: {path.name}")
    return payload


def _load_json(payload: bytes, *, label: str):
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TokenizerError(f"invalid {label} JSON") from exc


def verify_pipeline_artifacts(directory: Path) -> dict:
    """Verify report seal, safe inventory, shard hashes, rows, and coverage."""
    if not isinstance(directory, Path):
        raise TypeError("dataset directory must be pathlib.Path")
    report = _load_json(
        _read_bounded(directory / "manifest.json", 4 * 1024 * 1024),
        label="manifest",
    )
    expected = {
        "schema_version",
        "source",
        "config",
        "counts",
        "quality_rejections",
        "duplicates",
        "partitions",
        "tokenizer",
        "evidence_boundaries",
        "report_sha256",
    }
    if (
        not isinstance(report, dict)
        or set(report) != expected
        or type(report["schema_version"]) is not int
        or report["schema_version"] != 1
        or not isinstance(report["report_sha256"], str)
        or len(report["report_sha256"]) != 64
    ):
        raise TokenizerError("dataset manifest fields are invalid")
    sealed = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256(canonical_json_bytes(sealed)).hexdigest() != report["report_sha256"]:
        raise TokenizerError("dataset manifest seal mismatch")
    partitions = report["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation"}:
        raise TokenizerError("dataset partitions are invalid")
    expected_files = {"manifest.json"}
    seen_ids = set()
    for partition_name, partition in partitions.items():
        if not isinstance(partition, dict) or set(partition) != {"documents", "shards"}:
            raise TokenizerError("dataset partition fields are invalid")
        shards = partition["shards"]
        if not isinstance(shards, list) or not shards:
            raise TokenizerError("dataset partition requires nonempty shards")
        document_count = 0
        for index, descriptor in enumerate(shards):
            if not isinstance(descriptor, dict) or set(descriptor) != {
                "path",
                "documents",
                "utf8_text_bytes",
                "artifact_bytes",
                "sha256",
            }:
                raise TokenizerError("dataset shard descriptor fields are invalid")
            name = descriptor["path"]
            expected_name = f"{partition_name}-{index:05d}.jsonl"
            if (
                not isinstance(name, str)
                or name != expected_name
                or Path(name).name != name
            ):
                raise TokenizerError("dataset shard path is invalid")
            expected_files.add(name)
            artifact_bytes = descriptor["artifact_bytes"]
            if (
                type(artifact_bytes) is not int
                or not 1 <= artifact_bytes <= 16 * 1024 * 1024
            ):
                raise TokenizerError("dataset shard byte budget is invalid")
            payload = _read_bounded(directory / name, artifact_bytes)
            if (
                len(payload) != artifact_bytes
                or sha256(payload).hexdigest() != descriptor["sha256"]
            ):
                raise TokenizerError("dataset shard content identity mismatch")
            if not payload.endswith(b"\n"):
                raise TokenizerError("dataset shard must end with a newline")
            rows = [
                _load_json(line, label="shard row") for line in payload.splitlines()
            ]
            if len(rows) != descriptor["documents"] or not rows:
                raise TokenizerError("dataset shard document count mismatch")
            text_bytes = 0
            for row in rows:
                if not isinstance(row, dict) or set(row) != {
                    "document_id",
                    "source",
                    "source_sha256",
                    "text",
                }:
                    raise TokenizerError("dataset shard row fields are invalid")
                identity = row["document_id"]
                if (
                    not isinstance(identity, str)
                    or not identity
                    or identity in seen_ids
                    or not isinstance(row["text"], str)
                    or not row["text"]
                    or not isinstance(row["source"], str)
                    or not row["source"]
                    or not isinstance(row["source_sha256"], str)
                    or len(row["source_sha256"]) != 64
                ):
                    raise TokenizerError("dataset shard row values are invalid")
                seen_ids.add(identity)
                text_bytes += len(row["text"].encode("utf-8"))
            if text_bytes != descriptor["utf8_text_bytes"]:
                raise TokenizerError("dataset shard text byte count mismatch")
            document_count += len(rows)
        if document_count != partition["documents"]:
            raise TokenizerError("dataset partition document count mismatch")
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise TokenizerError("dataset artifact inventory mismatch")
    return report


def _partition_texts(directory: Path, report: dict, name: str) -> tuple[str, ...]:
    texts = []
    source_url = report["source"]["source_url"]
    for descriptor in report["partitions"][name]["shards"]:
        payload = _read_bounded(
            directory / descriptor["path"], descriptor["artifact_bytes"]
        )
        for line in payload.splitlines():
            row = _load_json(line, label="shard row")
            if row["source"] != source_url:
                raise TokenizerError("dataset row source does not match manifest")
            texts.append(row["text"])
    return tuple(texts)


def replay_pipeline_artifacts(directory: Path) -> dict:
    """Recompute tokenizer observations from verified published shard text."""
    report = verify_pipeline_artifacts(directory)
    tokenizer_payload = report["tokenizer"]
    if not isinstance(tokenizer_payload, dict) or set(tokenizer_payload) != {
        "state",
        "report",
    }:
        raise TokenizerError("dataset tokenizer evidence fields are invalid")
    tokenizer = tokenizer_from_state(tokenizer_payload["state"])
    train = compression_metrics(tokenizer, _partition_texts(directory, report, "train"))
    validation = compression_metrics(
        tokenizer, _partition_texts(directory, report, "validation")
    )
    total_tokens = train.tokens + validation.tokens
    total_bytes = train.utf8_bytes + validation.utf8_bytes
    actual = {
        "vocabulary_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "train": train.as_dict(),
        "validation": validation.as_dict(),
        "total_tokens": total_tokens,
        "total_utf8_bytes": total_bytes,
        "bytes_per_token": total_bytes / total_tokens,
        "token_reduction_fraction": 1 - total_tokens / total_bytes,
    }
    if actual != tokenizer_payload["report"]:
        raise TokenizerError("dataset tokenization evidence does not replay")
    return actual
