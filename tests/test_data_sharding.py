import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_cleaning import clean_documents
from ai_journey.data_documents import DataDocument
from ai_journey.data_sharding import shard_documents


def _documents(count=10):
    return clean_documents(
        tuple(
            DataDocument(str(index), f"document text {index}", "public://snapshot")
            for index in range(count)
        )
    )


def test_shards_are_stable_balanced_and_cover_each_document_once():
    documents = _documents()

    first = shard_documents(documents, shard_count=3)
    second = shard_documents(documents[::-1], shard_count=3)

    assert first == second
    sizes = [len(shard.documents) for shard in first]
    assert max(sizes) - min(sizes) == 1
    assert sorted(
        row.document_id for shard in first for row in shard.documents
    ) == sorted(row.document_id for row in documents)
    assert all(
        shard.utf8_bytes == sum(len(row.text.encode()) for row in shard.documents)
        for shard in first
    )


def test_shard_seed_changes_placement_but_not_coverage():
    first = shard_documents(_documents(), shard_count=3, seed=1)
    second = shard_documents(_documents(), shard_count=3, seed=2)

    assert first != second
    assert {row.document_id for shard in first for row in shard.documents} == {
        row.document_id for shard in second for row in shard.documents
    }


@pytest.mark.parametrize("count", [0, 11])
def test_shard_count_must_leave_nonempty_shards(count):
    with pytest.raises(TokenizerError, match="shard count"):
        shard_documents(_documents(), shard_count=count)
