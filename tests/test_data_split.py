import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_cleaning import clean_documents
from ai_journey.data_documents import DataDocument
from ai_journey.data_split import split_documents


def _documents(count=12):
    return clean_documents(
        tuple(
            DataDocument(str(index), f"unique document {index}", "public://snapshot")
            for index in range(count)
        )
    )


def test_split_is_stable_disjoint_and_independent_of_source_order():
    documents = _documents()

    first = split_documents(documents, validation_count=3, seed=59)
    second = split_documents(documents[::-1], validation_count=3, seed=59)

    assert first == second
    assert len(first.train) == 9
    assert len(first.validation) == 3
    assert {row.document_id for row in first.train}.isdisjoint(
        row.document_id for row in first.validation
    )


def test_split_seed_changes_assignment_without_changing_coverage():
    documents = _documents()
    first = split_documents(documents, validation_count=3, seed=1)
    second = split_documents(documents, validation_count=3, seed=2)

    assert first.validation != second.validation
    assert {row.document_id for row in first.train + first.validation} == {
        row.document_id for row in second.train + second.validation
    }


def test_split_rejects_cleaned_content_leakage():
    documents = clean_documents(
        (
            DataDocument("a", "same text", "source"),
            DataDocument("b", "same text", "source"),
        )
    )

    with pytest.raises(TokenizerError, match="unique ids and cleaned content"):
        split_documents(documents, validation_count=1)


@pytest.mark.parametrize("validation_count", [0, 2, 3])
def test_split_requires_nonempty_partitions(validation_count):
    with pytest.raises(TokenizerError, match="validation count"):
        split_documents(_documents(2), validation_count=validation_count)
