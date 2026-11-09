import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.minhash_assessment import CandidateAssessment
from ai_journey.minhash_clusters import duplicate_clusters


def _edge(left, right, duplicate=True):
    return CandidateAssessment(left, right, 1.0, 1.0, duplicate)


def test_duplicate_clusters_close_transitive_edges_deterministically():
    assessments = (_edge("b", "c"), _edge("a", "b"), _edge("c", "d", False))

    assert duplicate_clusters(("d", "c", "b", "a", "x"), assessments) == (
        ("a", "b", "c"),
    )


def test_duplicate_clusters_can_be_empty():
    assert duplicate_clusters(("a", "b"), (_edge("a", "b", False),)) == ()


@pytest.mark.parametrize("identities", [(), [], ("a", "a"), ("",)])
def test_duplicate_clusters_require_unique_document_ids(identities):
    with pytest.raises(TokenizerError, match="unique document ids"):
        duplicate_clusters(identities, ())


def test_duplicate_clusters_reject_unknown_assessment_documents():
    with pytest.raises(TokenizerError, match="unknown document"):
        duplicate_clusters(("a",), (_edge("a", "b"),))


def test_duplicate_clusters_require_assessment_records():
    with pytest.raises(TokenizerError, match="candidate assessments"):
        duplicate_clusters(("a",), (object(),))
