import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.minhash_candidates import candidate_pairs
from ai_journey.minhash_lsh import LSHBanding


def test_candidate_pairs_are_unique_sorted_and_order_independent():
    banding = LSHBanding(16, 4)
    shared = tuple(range(16))
    changed_last_band = tuple(range(12)) + (100, 101, 102, 103)
    isolated = tuple(range(1000, 1016))
    signatures = {"z": shared, "a": shared, "m": changed_last_band, "x": isolated}

    first = candidate_pairs(signatures, banding)
    second = candidate_pairs(dict(reversed(tuple(signatures.items()))), banding)

    assert first == second == (("a", "m"), ("a", "z"), ("m", "z"))


def test_candidate_pairs_can_return_an_empty_set():
    assert (
        candidate_pairs(
            {"a": tuple(range(16)), "b": tuple(range(100, 116))}, LSHBanding(16, 4)
        )
        == ()
    )


@pytest.mark.parametrize("signatures", [{}, [], {"": (0,) * 16}, {"bad id": (0,) * 16}])
def test_candidate_pairs_require_valid_document_ids(signatures):
    with pytest.raises(TokenizerError, match="signature ids"):
        candidate_pairs(signatures, LSHBanding(16, 4))


def test_candidate_pairs_require_banding_contract():
    with pytest.raises(TypeError, match="LSHBanding"):
        candidate_pairs({"a": (0,) * 16}, object())
