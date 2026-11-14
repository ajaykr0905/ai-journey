"""Seeded universal-hash families for reproducible MinHash signatures."""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256

from .bpe import TokenizerError


MERSENNE_PRIME = (1 << 61) - 1
_DOMAIN = b"ai-journey-minhash-family-v1\0"


@dataclass(frozen=True)
class MinHashFamily:
    """Seeded affine hashes (a * x + b) mod 2**61 - 1, one per signature slot.

    Coefficients come from SHA-256 of the seed rather than a runtime PRNG, so
    signatures match across processes; 16-512 hashes and a 64-bit seed.
    """

    num_hashes: int
    seed: int

    def __post_init__(self) -> None:
        if type(self.num_hashes) is not int or not 16 <= self.num_hashes <= 512:
            raise TokenizerError("MinHash family requires 16 to 512 hashes")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise TokenizerError("MinHash seed must be an unsigned 64-bit integer")

    def permutations(self) -> tuple[tuple[int, int], ...]:
        """Derive unique affine coefficients without runtime PRNG state.

        Every document signature asks for the same family, so the derivation is
        cached per (num_hashes, seed) and the immutable tuple is shared.
        """
        return _affine_coefficients(self.num_hashes, self.seed)


@lru_cache(maxsize=16)
def _affine_coefficients(num_hashes: int, seed: int) -> tuple[tuple[int, int], ...]:
    coefficients = []
    seen = set()
    index = 0
    seed_bytes = seed.to_bytes(8, "big")
    while len(coefficients) < num_hashes:
        digest = sha256(_DOMAIN + seed_bytes + index.to_bytes(8, "big")).digest()
        coefficient = (
            1 + int.from_bytes(digest[:8], "big") % (MERSENNE_PRIME - 1),
            int.from_bytes(digest[8:16], "big") % MERSENNE_PRIME,
        )
        if coefficient not in seen:
            seen.add(coefficient)
            coefficients.append(coefficient)
        index += 1
    return tuple(coefficients)
