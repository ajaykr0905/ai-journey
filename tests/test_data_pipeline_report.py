from dataclasses import replace
from hashlib import sha256

from ai_journey.data_documents import DataDocument
from ai_journey.data_pipeline import DataPipelineConfig, build_data_pipeline
from ai_journey.data_pipeline_report import (
    build_pipeline_report,
    canonical_json_bytes,
    shard_payload,
)
from ai_journey.data_pipeline_source import SourceEvidence
from ai_journey.data_quality import QualityPolicy
from ai_journey.tokenizer_sources import ENGLISH


def _evidence():
    documents = tuple(
        DataDocument(str(index), f"Useful public sentence number {index}.", ENGLISH.url)
        for index in range(9)
    )
    source = SourceEvidence(
        replace(ENGLISH, size_bytes=1).__dict__,
        ENGLISH.url,
        ENGLISH.license_url,
        documents,
    )
    config = DataPipelineConfig(
        quality=QualityPolicy(min_characters=5),
        validation_count=2,
        train_shards=2,
        validation_shards=1,
        vocabulary_size=280,
    )
    return source, build_data_pipeline(documents, config)


def test_report_seals_source_partitions_token_counts_and_evidence_boundaries():
    source, result = _evidence()

    report = build_pipeline_report(result, source)

    seal = report.pop("report_sha256")
    assert seal == sha256(canonical_json_bytes(report)).hexdigest()
    assert report["source"]["metadata"]["revision"] == ENGLISH.revision
    assert report["partitions"]["train"]["documents"] == 7
    assert report["partitions"]["validation"]["documents"] == 2
    assert report["tokenizer"]["report"]["total_tokens"] > 0
    assert report["evidence_boundaries"]["gpu_execution"] is False


def test_shard_payload_is_canonical_jsonl_with_matching_descriptor_digest():
    source, result = _evidence()
    report = build_pipeline_report(result, source)
    payload = shard_payload(result.train_shards[0])
    descriptor = report["partitions"]["train"]["shards"][0]

    assert payload.endswith(b"\n")
    assert descriptor["artifact_bytes"] == len(payload)
    assert descriptor["sha256"] == sha256(payload).hexdigest()
