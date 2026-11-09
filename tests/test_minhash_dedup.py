import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_cleaning import clean_documents
from ai_journey.data_documents import DataDocument
from ai_journey.minhash_dedup import MinHashDedupPolicy, deduplicate_with_minhash


def _documents(*rows):
    return clean_documents(
        tuple(
            DataDocument(identity, text, f"public://{identity}")
            for identity, text in rows
        )
    )


def _policy(**overrides):
    values = {
        "shingle_width": 2,
        "num_hashes": 64,
        "seed": 20261109,
        "bands": 32,
        "duplicate_threshold": 0.75,
    }
    values.update(overrides)
    return MinHashDedupPolicy(**values)


def test_minhash_dedup_removes_near_duplicates_and_reports_fraction():
    documents = _documents(
        ("b", "alpha beta gamma delta extra"),
        ("a", "alpha beta gamma delta"),
        ("c", "unrelated public corpus record"),
    )

    result = deduplicate_with_minhash(documents, _policy())

    assert [row.document_id for row in result.documents] == ["a", "c"]
    assert result.decisions[0].canonical_id == "a"
    assert result.decisions[0].duplicate_ids == ("b",)
    assert result.metrics.input_documents == 3
    assert result.metrics.candidate_pairs >= 1
    assert result.metrics.duplicate_pairs == 1
    assert result.metrics.removed_documents == 1
    assert result.metrics.removed_fraction == pytest.approx(1 / 3)


def test_minhash_dedup_is_independent_of_input_order():
    documents = _documents(("b", "same words here"), ("a", "same words here"))

    assert deduplicate_with_minhash(documents, _policy()) == deduplicate_with_minhash(
        documents[::-1], _policy()
    )


def test_minhash_dedup_preserves_corpus_without_qualified_duplicates():
    result = deduplicate_with_minhash(
        _documents(("a", "alpha beta"), ("b", "totally different")), _policy()
    )

    assert [row.document_id for row in result.documents] == ["a", "b"]
    assert result.metrics.removed_documents == 0
    assert result.metrics.removed_fraction == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"shingle_width": 0},
        {"num_hashes": 15},
        {"bands": 3},
        {"duplicate_threshold": 0},
    ],
)
def test_minhash_policy_rejects_invalid_cross_field_values(overrides):
    with pytest.raises(TokenizerError):
        _policy(**overrides)


def test_minhash_dedup_rejects_duplicate_document_ids():
    documents = _documents(("a", "one text"), ("a", "other text"))
    with pytest.raises(TokenizerError, match="unique document ids"):
        deduplicate_with_minhash(documents, _policy())
