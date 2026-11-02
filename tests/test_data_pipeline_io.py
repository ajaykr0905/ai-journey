from dataclasses import replace
import json
import os

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_documents import DataDocument
from ai_journey.data_pipeline import DataPipelineConfig, build_data_pipeline
from ai_journey.data_pipeline_io import publish_pipeline_artifacts
from ai_journey.data_pipeline_source import SourceEvidence
from ai_journey.data_quality import QualityPolicy
from ai_journey.tokenizer_sources import ENGLISH


def _build():
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
    config = DataPipelineConfig(
        quality=QualityPolicy(min_characters=5),
        validation_count=2,
        train_shards=2,
        vocabulary_size=280,
    )
    return source, build_data_pipeline(documents, config)


def test_publication_exposes_complete_shards_and_manifest_together(tmp_path):
    source, result = _build()
    output = tmp_path / "dataset"

    report = publish_pipeline_artifacts(output, result, source)

    assert sorted(path.name for path in output.iterdir()) == [
        "manifest.json",
        "train-00000.jsonl",
        "train-00001.jsonl",
        "validation-00000.jsonl",
    ]
    assert json.loads((output / "manifest.json").read_bytes()) == report
    for partition in report["partitions"].values():
        assert all((output / shard["path"]).is_file() for shard in partition["shards"])


def test_publication_failure_leaves_no_visible_or_temporary_dataset(
    tmp_path, monkeypatch
):
    source, result = _build()
    output = tmp_path / "dataset"

    def fail_replace(_source, _destination):
        raise OSError("injected publication failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        publish_pipeline_artifacts(output, result, source)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_publication_never_overwrites_an_existing_dataset(tmp_path):
    source, result = _build()
    output = tmp_path / "dataset"
    output.mkdir()
    marker = output / "preserve"
    marker.write_text("user data")

    with pytest.raises(TokenizerError, match="already exists"):
        publish_pipeline_artifacts(output, result, source)

    assert marker.read_text() == "user data"
