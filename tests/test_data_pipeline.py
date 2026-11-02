import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_documents import DataDocument
from ai_journey.data_pipeline import (
    DataPipelineConfig,
    build_data_pipeline,
)
from ai_journey.data_quality import QualityPolicy


def _documents():
    rows = [
        DataDocument(str(index), f"Useful public sentence number {index}.", "source")
        for index in range(10)
    ]
    rows.extend(
        [
            DataDocument("duplicate", rows[0].text, "source"),
            DataDocument("short", "no", "source"),
            DataDocument("spacing", "Useful   spaced\n sentence.", "source"),
        ]
    )
    return tuple(rows)


def _config():
    return DataPipelineConfig(
        quality=QualityPolicy(min_characters=10),
        validation_count=3,
        train_shards=2,
        validation_shards=1,
        vocabulary_size=280,
    )


def test_pipeline_compounds_cleaning_quality_dedup_split_shards_and_tokens():
    result = build_data_pipeline(_documents(), _config())

    assert result.input_documents == 13
    assert result.changed_documents == 1
    assert [
        decision.document.document_id
        for decision in result.quality
        if not decision.accepted
    ] == ["short"]
    assert len(result.deduplication.duplicates) == 1
    assert len(result.split.validation) == 3
    assert len(result.split.train) == 8
    assert len(result.train_shards) == 2
    assert result.tokenization.total_tokens > 0
    assert result.tokenization.total_utf8_bytes > result.tokenization.total_tokens


def test_pipeline_is_deterministic_under_input_reordering():
    first = build_data_pipeline(_documents(), _config())
    second = build_data_pipeline(_documents()[::-1], _config())

    assert first.deduplication == second.deduplication
    assert first.split == second.split
    assert first.train_shards == second.train_shards
    assert first.tokenization == second.tokenization


def test_pipeline_fails_when_quality_removes_split_capacity():
    documents = tuple(DataDocument(str(index), "tiny", "source") for index in range(5))

    with pytest.raises(TokenizerError, match="quality admission"):
        build_data_pipeline(documents, _config())
