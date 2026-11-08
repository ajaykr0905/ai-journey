import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.minhash_family import MERSENNE_PRIME, MinHashFamily


def test_family_derives_reproducible_unique_affine_coefficients():
    family = MinHashFamily(num_hashes=16, seed=20261109)

    first = family.permutations()
    second = family.permutations()

    assert first == second
    assert len(first) == len(set(first)) == 16
    assert all(1 <= coefficient < MERSENNE_PRIME for coefficient, _ in first)
    assert all(0 <= offset < MERSENNE_PRIME for _, offset in first)


def test_family_seed_changes_the_permutation_contract():
    assert MinHashFamily(16, 1).permutations() != MinHashFamily(16, 2).permutations()


@pytest.mark.parametrize("count", [True, 15, 513, 16.0])
def test_family_rejects_invalid_hash_counts(count):
    with pytest.raises(TokenizerError, match="16 to 512"):
        MinHashFamily(count, 1)


@pytest.mark.parametrize("seed", [True, -1, 2**64, 1.0])
def test_family_rejects_invalid_seeds(seed):
    with pytest.raises(TokenizerError, match="seed"):
        MinHashFamily(16, seed)
