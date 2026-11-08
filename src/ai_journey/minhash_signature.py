"""MinHash signature construction over stable shingle hashes."""

from .bpe import TokenizerError
from .minhash_family import MERSENNE_PRIME, MinHashFamily


def build_minhash_signature(
    shingle_hashes: tuple[int, ...], family: MinHashFamily
) -> tuple[int, ...]:
    """Return the minimum affine hash for every permutation in the family."""
    if (
        not isinstance(shingle_hashes, tuple)
        or not shingle_hashes
        or any(
            type(value) is not int or not 0 <= value < 2**64 for value in shingle_hashes
        )
    ):
        raise TokenizerError("MinHash signature requires unsigned 64-bit hashes")
    if not isinstance(family, MinHashFamily):
        raise TypeError("MinHash signature requires a MinHashFamily")
    reduced = tuple({value % MERSENNE_PRIME for value in shingle_hashes})
    return tuple(
        min((coefficient * value + offset) % MERSENNE_PRIME for value in reduced)
        for coefficient, offset in family.permutations()
    )


def signature_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """Estimate Jaccard similarity as the fraction of matching minima."""
    if (
        not isinstance(left, tuple)
        or not isinstance(right, tuple)
        or not left
        or len(left) != len(right)
        or any(
            type(value) is not int or not 0 <= value < MERSENNE_PRIME
            for value in (*left, *right)
        )
    ):
        raise TokenizerError("MinHash signatures must be equal nonempty hash tuples")
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
