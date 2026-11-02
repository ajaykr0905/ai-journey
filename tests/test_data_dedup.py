import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_cleaning import clean_documents
from ai_journey.data_dedup import deduplicate_documents
from ai_journey.data_documents import DataDocument


def _clean(*rows):
    return clean_documents(
        tuple(
            DataDocument(identity, text, "public://snapshot") for identity, text in rows
        )
    )


def test_deduplication_keeps_stable_canonical_ids_independent_of_input_order():
    documents = _clean(("z", "same text"), ("a", "same text"), ("b", "unique"))

    first = deduplicate_documents(documents)
    second = deduplicate_documents(documents[::-1])

    assert first == second
    assert [row.document_id for row in first.documents] == ["a", "b"]
    assert [(row.duplicate_id, row.canonical_id) for row in first.duplicates] == [
        ("z", "a")
    ]
    assert len(first.duplicates[0].content_sha256) == 64


def test_deduplication_uses_cleaned_text_identity():
    result = deduplicate_documents(_clean(("a", "Cafe\u0301"), ("b", "Café")))

    assert [row.document_id for row in result.documents] == ["a"]
    assert result.duplicates[0].duplicate_id == "b"


def test_deduplication_rejects_duplicate_record_ids():
    with pytest.raises(TokenizerError, match="unique"):
        deduplicate_documents(_clean(("a", "one"), ("a", "two")))
