import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.minhash_family import MERSENNE_PRIME, MinHashFamily
from ai_journey.minhash_signature import build_minhash_signature


def test_signature_has_one_bounded_minimum_per_permutation():
    family = MinHashFamily(16, 17)
    signature = build_minhash_signature((10, 20, 30), family)

    assert len(signature) == 16
    assert all(0 <= value < MERSENNE_PRIME for value in signature)
    assert signature == tuple(
        min((a * value + b) % MERSENNE_PRIME for value in (10, 20, 30))
        for a, b in family.permutations()
    )


def test_signature_is_independent_of_hash_order_and_duplicates():
    family = MinHashFamily(32, 3)
    assert build_minhash_signature((3, 2, 1, 2), family) == build_minhash_signature(
        (1, 2, 3), family
    )


@pytest.mark.parametrize("values", [(), [], (-1,), (2**64,), (1.0,)])
def test_signature_rejects_invalid_hash_inputs(values):
    with pytest.raises(TokenizerError, match="unsigned 64-bit"):
        build_minhash_signature(values, MinHashFamily(16, 1))


def test_signature_requires_a_family_contract():
    with pytest.raises(TypeError, match="MinHashFamily"):
        build_minhash_signature((1,), object())
