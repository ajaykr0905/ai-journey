from hashlib import sha256

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_cleaning import clean_document, clean_documents
from ai_journey.data_documents import DataDocument


def test_cleaning_normalizes_unicode_and_whitespace_with_source_identity():
    source = " Cafe\u0301\n\ttext  "

    cleaned = clean_document(DataDocument("a", source, "public://snapshot"))

    assert cleaned.text == "Café text"
    assert cleaned.changed is True
    assert cleaned.source_sha256 == sha256(source.encode()).hexdigest()
    assert cleaned.document_id == "a"


def test_cleaning_marks_unchanged_text_without_rewriting_identity():
    documents = (DataDocument("a", "Already clean.", "public://snapshot"),)

    cleaned = clean_documents(documents)

    assert cleaned[0].text == "Already clean."
    assert cleaned[0].changed is False


@pytest.mark.parametrize("text", ["bad\x00text", "bad\x01text", "bad\ufffdtext"])
def test_cleaning_rejects_ambiguous_or_corrupt_text(text):
    with pytest.raises(TokenizerError, match="control|replacement"):
        clean_document(DataDocument("a", text, "public://snapshot"))
