"""Locality-sensitive hashing parameters for MinHash candidate discovery."""

from dataclasses import dataclass
import math

from .bpe import TokenizerError


@dataclass(frozen=True)
class LSHBanding:
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
