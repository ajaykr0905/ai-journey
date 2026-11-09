"""Exact similarity controls for approximate MinHash candidates."""

from .bpe import TokenizerError


def exact_jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    """Compute exact set Jaccard similarity for two nonempty shingle tuples."""
    for value in (left, right):
        if (
            not isinstance(value, tuple)
            or not value
            or any(not isinstance(shingle, str) or not shingle for shingle in value)
        ):
            raise TokenizerError("Jaccard similarity requires nonempty shingle tuples")
    left_set = set(left)
    right_set = set(right)
    return len(left_set & right_set) / len(left_set | right_set)
