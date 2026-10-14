"""Independent byte-pair tokenization primitives with deterministic tie breaking."""

from collections import Counter
from collections.abc import Sequence


class TokenizerError(ValueError):
    """Invalid tokenizer inputs or serialized state."""


def token_ids(values: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError("token ids must be a list or tuple")
    if any(type(value) is not int or value < 0 for value in values):
        raise TokenizerError("token ids must be nonnegative integers")
    return tuple(values)


def pair_counts(values: Sequence[int]) -> Counter:
    """Count adjacent pairs, including overlaps, without mutating the stream."""
    values = token_ids(values)
    return Counter(zip(values, values[1:]))


def merge_pair(
    values: Sequence[int], pair: tuple[int, int], replacement: int
) -> tuple[int, ...]:
    """Replace nonoverlapping matches left to right."""
    values = token_ids(values)
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise TokenizerError("pair must contain two token ids")
    token_ids(list(pair) + [replacement])
    if replacement in pair:
        raise TokenizerError("replacement must be a new token id")
    output = []
    index = 0
    while index < len(values):
        if values[index : index + 2] == pair:
            output.append(replacement)
            index += 2
        else:
            output.append(values[index])
            index += 1
    return tuple(output)
