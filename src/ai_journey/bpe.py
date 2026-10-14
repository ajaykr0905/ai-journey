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


class ByteTokenizer:
    """Lossless UTF-8 byte BPE with immutable, dependency-ordered merge ranks."""

    def __init__(self, merges: tuple[tuple[int, int], ...] = ()):
        if not isinstance(merges, tuple) or len(merges) > 3840:
            raise TokenizerError("merges must be a tuple of at most 3840 pairs")
        vocabulary = [bytes([value]) for value in range(256)]
        seen = set()
        for pair in merges:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TokenizerError("each merge must contain two prior token ids")
            token_ids(list(pair))
            if pair in seen or any(value >= len(vocabulary) for value in pair):
                raise TokenizerError(
                    "merge ranks must be unique and dependency ordered"
                )
            seen.add(pair)
            vocabulary.append(vocabulary[pair[0]] + vocabulary[pair[1]])
        self._merges = merges
        self._vocabulary = tuple(vocabulary)
        self._ranks = {pair: index for index, pair in enumerate(merges)}

    @property
    def merges(self) -> tuple[tuple[int, int], ...]:
        return self._merges

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    @classmethod
    def train(cls, text: str, *, vocab_size: int = 320, min_frequency: int = 2):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if type(vocab_size) is not int or not 256 <= vocab_size <= 4096:
            raise TokenizerError("vocab_size must be in [256, 4096]")
        if type(min_frequency) is not int or min_frequency < 2:
            raise TokenizerError("min_frequency must be at least two")
        data = text.encode("utf-8")
        if len(data) > 1024 * 1024:
            raise TokenizerError("training corpus exceeds 1 MiB")
        values = tuple(data)
        merges = []
        while len(merges) + 256 < vocab_size:
            counts = pair_counts(values)
            if not counts:
                break
            pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
            if counts[pair] < min_frequency:
                break
            values = merge_pair(values, pair, 256 + len(merges))
            merges.append(pair)
        return cls(tuple(merges))

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        values = tuple(text.encode("utf-8"))
        while len(values) > 1:
            available = set(pair_counts(values)) & self._ranks.keys()
            if not available:
                break
            pair = min(available, key=self._ranks.__getitem__)
            values = merge_pair(values, pair, 256 + self._ranks[pair])
        return values

    def decode_bytes(self, values) -> bytes:
        values = token_ids(values)
        if any(value >= self.vocab_size for value in values):
            raise TokenizerError("token id is outside the vocabulary")
        return b"".join(self._vocabulary[value] for value in values)

    def decode(self, values) -> str:
        return self.decode_bytes(values).decode("utf-8")
