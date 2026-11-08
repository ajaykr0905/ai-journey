import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.minhash_similarity import exact_jaccard


def test_exact_jaccard_uses_set_overlap():
    assert exact_jaccard(("a", "b", "b"), ("b", "c")) == pytest.approx(1 / 3)
    assert exact_jaccard(("a", "b"), ("b", "a")) == 1.0
    assert exact_jaccard(("a",), ("b",)) == 0.0


@pytest.mark.parametrize(
    ("left", "right"),
    [((), ("a",)), ([], ("a",)), (("",), ("a",)), (("a",), (1,))],
)
def test_exact_jaccard_requires_nonempty_text_shingles(left, right):
    with pytest.raises(TokenizerError, match="shingle tuples"):
        exact_jaccard(left, right)
