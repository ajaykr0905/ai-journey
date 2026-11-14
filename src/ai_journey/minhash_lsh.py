"""Locality-sensitive hashing parameters for MinHash candidate discovery."""

from dataclasses import dataclass
from hashlib import sha256
import math

from .bpe import TokenizerError
from .minhash_family import MERSENNE_PRIME


_BAND_DOMAIN = b"ai-journey-minhash-band-v1\0"


@dataclass(frozen=True)
class LSHBanding:
    """Equal signature bands; documents that share any band become candidates.

    bands must divide num_hashes; more bands lower the similarity at which
    pairs start to collide.
    """

    num_hashes: int
    bands: int

    def __post_init__(self) -> None:
        if type(self.num_hashes) is not int or not 16 <= self.num_hashes <= 512:
            raise TokenizerError("LSH banding requires 16 to 512 hashes")
        if (
            type(self.bands) is not int
            or not 1 <= self.bands <= self.num_hashes
            or self.num_hashes % self.bands
        ):
            raise TokenizerError("LSH bands must divide the signature length")

    @property
    def rows_per_band(self) -> int:
        """Signature values per band; a band matches with probability similarity**r."""
        return self.num_hashes // self.bands

    @property
    def approximate_threshold(self) -> float:
        """Return the conventional collision-curve inflection estimate."""
        return (1 / self.bands) ** (1 / self.rows_per_band)

    def collision_probability(self, similarity: float) -> float:
        """Return the probability that at least one band matches."""
        if (
            isinstance(similarity, bool)
            or not isinstance(similarity, (int, float))
            or not math.isfinite(similarity)
            or not 0 <= similarity <= 1
        ):
            raise TokenizerError("LSH similarity must be finite from zero to one")
        return 1 - (1 - float(similarity) ** self.rows_per_band) ** self.bands


def band_keys(
    signature: tuple[int, ...], banding: LSHBanding
) -> tuple[tuple[int, str], ...]:
    """Encode each signature band as a domain-separated bucket identity."""
    if not isinstance(banding, LSHBanding):
        raise TypeError("LSH bucket keys require an LSHBanding contract")
    if (
        not isinstance(signature, tuple)
        or len(signature) != banding.num_hashes
        or any(
            type(value) is not int or not 0 <= value < MERSENNE_PRIME
            for value in signature
        )
    ):
        raise TokenizerError("LSH bucket signature does not match its banding")
    width = banding.rows_per_band
    keys = []
    for band in range(banding.bands):
        values = signature[band * width : (band + 1) * width]
        payload = b"".join(value.to_bytes(8, "big") for value in values)
        digest = sha256(_BAND_DOMAIN + band.to_bytes(4, "big") + payload).hexdigest()
        keys.append((band, digest))
    return tuple(keys)
