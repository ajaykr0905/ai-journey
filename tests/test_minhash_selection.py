import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.minhash_selection import ClusterDecision, select_representatives


def test_representatives_keep_lexicographically_first_ids():
    kept, decisions = select_representatives(
        ("z", "a", "b", "x", "m"), (("m", "x"), ("z", "a", "b"))
    )

    assert kept == ("a", "m")
    assert decisions == (
        ClusterDecision("a", ("b", "z")),
        ClusterDecision("m", ("x",)),
    )


def test_representatives_preserve_all_ids_without_clusters():
    assert select_representatives(("b", "a"), ()) == (("a", "b"), ())


@pytest.mark.parametrize("identities", [(), [], ("a", "a")])
def test_representatives_require_unique_document_ids(identities):
    with pytest.raises(TokenizerError, match="unique document ids"):
        select_representatives(identities, ())


@pytest.mark.parametrize(
    "clusters",
    [
        (("a",),),
        (("a", "a"),),
        (("a", "missing"),),
        (("a", "b"), ("b", "c")),
    ],
)
def test_representatives_reject_invalid_clusters(clusters):
    with pytest.raises(TokenizerError, match="disjoint known"):
        select_representatives(("a", "b", "c"), clusters)
