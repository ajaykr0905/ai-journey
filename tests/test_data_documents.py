import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_documents import DataDocument, documents_from_sentences
from ai_journey.parallel_corpus import Sentence


def test_sentence_documents_preserve_identity_text_and_source():
    records = (Sentence("s2", "two"), Sentence("s1", "one"))

    assert documents_from_sentences(records, source="public://snapshot") == (
        DataDocument("s2", "two", "public://snapshot"),
        DataDocument("s1", "one", "public://snapshot"),
    )


@pytest.mark.parametrize(
    ("document_id", "text", "source"),
    [
        ("", "text", "source"),
        ("bad id", "text", "source"),
        ("id", " ", "source"),
        ("id", "text", ""),
    ],
)
def test_document_rejects_ambiguous_identity_or_provenance(document_id, text, source):
    with pytest.raises(TokenizerError):
        DataDocument(document_id, text, source)


def test_document_collection_rejects_duplicate_ids():
    records = (Sentence("same", "one"), Sentence("same", "two"))

    with pytest.raises(TokenizerError, match="unique"):
        documents_from_sentences(records, source="public://snapshot")
