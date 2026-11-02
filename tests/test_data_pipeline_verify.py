from dataclasses import replace
from hashlib import sha256
import json

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_documents import DataDocument
from ai_journey.data_pipeline import DataPipelineConfig, build_data_pipeline
from ai_journey.data_pipeline_io import publish_pipeline_artifacts
from ai_journey.data_pipeline_report import canonical_json_bytes
from ai_journey.data_pipeline_source import SourceEvidence
from ai_journey.data_pipeline_verify import (
    replay_pipeline_artifacts,
    verify_pipeline_artifacts,
)
from ai_journey.data_quality import QualityPolicy
from ai_journey.tokenizer_sources import ENGLISH


def _publish(tmp_path):
    documents = tuple(
        DataDocument(str(index), f"Useful public sentence number {index}.", ENGLISH.url)
        for index in range(8)
    )
    source = SourceEvidence(
        replace(ENGLISH, size_bytes=1).__dict__,
        ENGLISH.url,
        ENGLISH.license_url,
        documents,
    )
    result = build_data_pipeline(
        documents,
        DataPipelineConfig(
            quality=QualityPolicy(min_characters=5),
            validation_count=2,
            train_shards=2,
            vocabulary_size=280,
        ),
    )
    output = tmp_path / "dataset"
    publish_pipeline_artifacts(output, result, source)
    return output


def test_verifier_accepts_complete_independently_read_artifacts(tmp_path):
    output = _publish(tmp_path)

    report = verify_pipeline_artifacts(output)

    assert report["partitions"]["train"]["documents"] == 6
    assert report["partitions"]["validation"]["documents"] == 2


def test_verifier_rejects_shard_tampering(tmp_path):
    output = _publish(tmp_path)
    path = output / "train-00000.jsonl"
    path.write_bytes(path.read_bytes().replace(b"sentence", b"tampered", 1))

    with pytest.raises(TokenizerError, match="content identity"):
        verify_pipeline_artifacts(output)


def test_verifier_rejects_resealed_unsafe_shard_paths(tmp_path):
    output = _publish(tmp_path)
    manifest_path = output / "manifest.json"
    report = json.loads(manifest_path.read_text())
    report["partitions"]["train"]["shards"][0]["path"] = "../escape.jsonl"
    unsealed = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = sha256(canonical_json_bytes(unsealed)).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(report))

    with pytest.raises(TokenizerError, match="path"):
        verify_pipeline_artifacts(output)


def test_verifier_rejects_unlisted_artifacts(tmp_path):
    output = _publish(tmp_path)
    (output / "unexpected").write_text("junk")

    with pytest.raises(TokenizerError, match="inventory"):
        verify_pipeline_artifacts(output)


def test_verifier_rejects_unlisted_directories(tmp_path):
    output = _publish(tmp_path)
    (output / "unexpected").mkdir()

    with pytest.raises(TokenizerError, match="inventory"):
        verify_pipeline_artifacts(output)


def test_verifier_rejects_shard_symlinks_even_when_target_bytes_match(tmp_path):
    output = _publish(tmp_path)
    shard = output / "train-00000.jsonl"
    external = tmp_path / "external.jsonl"
    shard.replace(external)
    shard.symlink_to(external)

    with pytest.raises(TokenizerError, match="regular file"):
        verify_pipeline_artifacts(output)


def test_replay_recomputes_token_counts_and_compression_from_shards(tmp_path):
    output = _publish(tmp_path)

    replay = replay_pipeline_artifacts(output)

    assert replay["total_tokens"] > 0
    assert replay["total_utf8_bytes"] > replay["total_tokens"]
    assert replay["bytes_per_token"] == (
        replay["total_utf8_bytes"] / replay["total_tokens"]
    )


def test_replay_rejects_resealed_token_observation_drift(tmp_path):
    output = _publish(tmp_path)
    manifest_path = output / "manifest.json"
    report = json.loads(manifest_path.read_text())
    report["tokenizer"]["report"]["total_tokens"] += 1
    unsealed = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = sha256(canonical_json_bytes(unsealed)).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(report))

    with pytest.raises(TokenizerError, match="does not replay"):
        replay_pipeline_artifacts(output)


@pytest.mark.parametrize(
    ("field", "value"),
    [("sha256", 7), ("artifact_bytes", True), ("documents", 0)],
)
def test_verifier_rejects_resealed_invalid_shard_descriptor_types(
    tmp_path, field, value
):
    output = _publish(tmp_path)
    manifest_path = output / "manifest.json"
    report = json.loads(manifest_path.read_text())
    report["partitions"]["train"]["shards"][0][field] = value
    unsealed = {key: item for key, item in report.items() if key != "report_sha256"}
    report["report_sha256"] = sha256(canonical_json_bytes(unsealed)).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(report))

    with pytest.raises(TokenizerError, match="byte budget"):
        verify_pipeline_artifacts(output)


def test_verifier_rejects_resealed_missing_public_attribution(tmp_path):
    output = _publish(tmp_path)
    manifest_path = output / "manifest.json"
    report = json.loads(manifest_path.read_text())
    report["source"]["attribution"] = ""
    unsealed = {key: item for key, item in report.items() if key != "report_sha256"}
    report["report_sha256"] = sha256(canonical_json_bytes(unsealed)).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(report))

    with pytest.raises(TokenizerError, match="source evidence"):
        verify_pipeline_artifacts(output)
