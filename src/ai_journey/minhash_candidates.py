"""Deterministic LSH candidate discovery from MinHash signatures."""

from itertools import combinations

from .bpe import TokenizerError
from .minhash_lsh import LSHBanding, band_keys


def candidate_pairs(
    signatures: dict[str, tuple[int, ...]], banding: LSHBanding
) -> tuple[tuple[str, str], ...]:
    """Return each document pair sharing at least one LSH bucket exactly once."""
    if not isinstance(banding, LSHBanding):
        raise TypeError("candidate discovery requires an LSHBanding contract")
    if (
        not isinstance(signatures, dict)
        or not signatures
        or any(
            not isinstance(document_id, str)
            or not document_id
            or any(character.isspace() for character in document_id)
            for document_id in signatures
        )
    ):
        raise TokenizerError("candidate discovery requires document signature ids")
    buckets: dict[tuple[int, str], list[str]] = {}
    for document_id in sorted(signatures):
        for key in band_keys(signatures[document_id], banding):
            buckets.setdefault(key, []).append(document_id)
    pairs = {
        pair
        for members in buckets.values()
        if len(members) > 1
        for pair in combinations(members, 2)
    }
    return tuple(sorted(pairs))
