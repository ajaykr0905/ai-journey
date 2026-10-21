import pytest
from ai_journey.ddp_evaluation import Document, documents_from_lines
from ai_journey.ddp_evaluation import split_documents


def test_public_lines_keep_individual_document_boundaries():
    documents = documents_from_lines(b"alice\n\nbob\n")
    assert [document.identifier for document in documents] == ["line-1", "line-3"]
    assert [document.content for document in documents] == [b"alice\n", b"bob\n"]
    assert len({document.digest for document in documents}) == 2


@pytest.mark.parametrize("source", [b"", b"alice\nalice\n"])
def test_empty_or_duplicated_public_documents_are_rejected(source):
    with pytest.raises(ValueError):
        documents_from_lines(source)


def test_one_byte_document_cannot_supply_a_next_token_target():
    with pytest.raises(ValueError):
        Document("short", b"x")


def test_document_split_is_disjoint_and_invariant_to_input_order():
    documents = documents_from_lines(b"alice\nbob\ncarol\ndave\n")
    train, valid = split_documents(documents, validation_count=2)
    assert not {doc.digest for doc in train} & {doc.digest for doc in valid}
    assert (train, valid) == split_documents(reversed(documents), validation_count=2)
    assert len(train) == len(valid) == 2
