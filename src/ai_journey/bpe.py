"""Independent byte-pair tokenization primitives with deterministic tie breaking."""

from collections import Counter
from collections.abc import Sequence
import re


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
        return cls._train_segments(
            (text,), vocab_size=vocab_size, min_frequency=min_frequency
        )

    @classmethod
    def _train_segments(cls, segments, *, vocab_size, min_frequency):
        streams = [tuple(segment.encode("utf-8")) for segment in segments]
        merges = []
        while len(merges) + 256 < vocab_size:
            counts = Counter()
            for values in streams:
                counts.update(pair_counts(values))
            if not counts:
                break
            pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
            if counts[pair] < min_frequency:
                break
            streams = [
                merge_pair(values, pair, 256 + len(merges)) for values in streams
            ]
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


DEFAULT_PATTERN = r"\w+|[^\w\s]+|\s+"


class RegexTokenizer(ByteTokenizer):
    """Experimental stdlib-Unicode segmentation; merges never cross regex pieces."""

    def __init__(self, merges=(), *, pattern=DEFAULT_PATTERN):
        super().__init__(merges)
        if not isinstance(pattern, str) or not pattern:
            raise TokenizerError("pattern must be a nonempty string")
        try:
            self._compiled = re.compile(pattern)
        except re.error as exc:
            raise TokenizerError("invalid tokenizer pattern") from exc
        if self._compiled.match("") is not None:
            raise TokenizerError("pattern must not match empty text")
        self.pattern = pattern

    def pieces(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        pieces = []
        position = 0
        for match in self._compiled.finditer(text):
            if match.start() != position or match.end() == position:
                raise TokenizerError("pattern must cover every character exactly once")
            pieces.append(match.group())
            position = match.end()
        if position != len(text):
            raise TokenizerError("pattern must cover every character exactly once")
        return tuple(pieces)

    @classmethod
    def train(cls, text, *, vocab_size=320, min_frequency=2, pattern=DEFAULT_PATTERN):
        # Reuse basic training validation before applying boundary-aware counts.
        ByteTokenizer.train("", vocab_size=vocab_size, min_frequency=min_frequency)
        splitter = cls(pattern=pattern)
        if len(text.encode("utf-8")) > 1024 * 1024:
            raise TokenizerError("training corpus exceeds 1 MiB")
        trained = ByteTokenizer._train_segments(
            splitter.pieces(text), vocab_size=vocab_size, min_frequency=min_frequency
        )
        return cls(trained.merges, pattern=pattern)

    def encode(self, text):
        return tuple(
            token
            for piece in self.pieces(text)
            for token in super(RegexTokenizer, self).encode(piece)
        )
