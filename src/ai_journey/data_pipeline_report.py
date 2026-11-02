"""Canonical manifest and shard payloads for verified dataset builds."""

from dataclasses import asdict
from hashlib import sha256
import json

from .data_pipeline import DataPipelineResult
from .data_pipeline_source import SourceEvidence
from .data_sharding import DataShard
from .tokenizer_io import tokenizer_state


def canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def shard_payload(shard: DataShard) -> bytes:
    """Encode one document per canonical JSON line in stable shard order."""
    return b"".join(
        canonical_json_bytes(
            {
                "document_id": row.document_id,
                "source": row.source,
                "source_sha256": row.source_sha256,
                "text": row.text,
            }
        )
        for row in shard.documents
    )


def _partition_payload(name: str, shards: tuple[DataShard, ...]) -> dict:
    descriptors = []
    for shard in shards:
        payload = shard_payload(shard)
        descriptors.append(
            {
                "path": f"{name}-{shard.index:05d}.jsonl",
                "documents": len(shard.documents),
                "utf8_text_bytes": shard.utf8_bytes,
                "artifact_bytes": len(payload),
                "sha256": sha256(payload).hexdigest(),
            }
        )
    return {
        "documents": sum(item["documents"] for item in descriptors),
        "shards": descriptors,
    }


def build_pipeline_report(result: DataPipelineResult, source: SourceEvidence) -> dict:
    """Describe all accepted, rejected, duplicate, split, and token outcomes."""
    if not isinstance(result, DataPipelineResult) or not isinstance(
        source, SourceEvidence
    ):
        raise TypeError("report requires pipeline and source evidence")
    report = {
        "schema_version": 1,
        "source": {
            "metadata": source.metadata,
            "source_url": source.source_url,
            "license_url": source.license_url,
        },
        "config": asdict(result.config),
        "counts": {
            "input_documents": result.input_documents,
            "changed_documents": result.changed_documents,
            "quality_rejected": sum(
                not decision.accepted for decision in result.quality
            ),
            "exact_duplicates": len(result.deduplication.duplicates),
        },
        "quality_rejections": [
            {
                "document_id": decision.document.document_id,
                "reasons": list(decision.reasons),
            }
            for decision in result.quality
            if not decision.accepted
        ],
        "duplicates": [asdict(row) for row in result.deduplication.duplicates],
        "partitions": {
            "train": _partition_payload("train", result.train_shards),
            "validation": _partition_payload("validation", result.validation_shards),
        },
        "tokenizer": {
            "state": tokenizer_state(result.tokenizer),
            "report": result.tokenization.as_dict(),
        },
        "evidence_boundaries": {
            "public_data": True,
            "cpu_processing": True,
            "gpu_execution": False,
            "performance_measurement": False,
        },
    }
    report["report_sha256"] = sha256(canonical_json_bytes(report)).hexdigest()
    return report
