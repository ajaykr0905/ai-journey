"""Stable content hashes for MinHash shingles."""

from hashlib import sha256

from .bpe import TokenizerError


_DOMAIN = b"ai-journey-minhash-shingle-v1\0"


def hash_shingle(shingle: str) -> int:
    """Map a nonempty Unicode shingle into a deterministic unsigned 64-bit value."""
    if not isinstance(shingle, str) or not shingle:
        raise TokenizerError("MinHash shingle must be nonempty text")
    digest = sha256(_DOMAIN + shingle.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def hash_shingles(shingles: tuple[str, ...]) -> tuple[int, ...]:
    """Hash a nonempty shingle set and fail closed on a truncated-hash collision."""
    if (
        not isinstance(shingles, tuple)
        or not shingles
        or not all(isinstance(shingle, str) and shingle for shingle in shingles)
    ):
        raise TokenizerError("MinHash shingles must be a nonempty text tuple")
    by_hash: dict[int, str] = {}
    for shingle in shingles:
        value = hash_shingle(shingle)
        previous = by_hash.setdefault(value, shingle)
        if previous != shingle:
            raise TokenizerError("MinHash shingle hash collision")
    return tuple(sorted(by_hash))
