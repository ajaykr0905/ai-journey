"""Independent byte-pair tokenization primitives with deterministic tie breaking."""

from collections import Counter
from collections.abc import Sequence
import re


class TokenizerError(ValueError):
    """Invalid tokenizer inputs or serialized state."""


def token_ids(values: Sequence[int]) -> tuple[int, ...]:
    """Validate an untrusted list or tuple of ids and return an immutable copy.

    bool, float and negative entries raise TokenizerError instead of aliasing
    real ids; upper bounds are checked by the callers that know the vocabulary.
    """
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
    return _merge_known_pair(values, pair, replacement)


def _merge_known_pair(values, pair, replacement):
    """Merge already-validated ids, jumping between C-level searches for pair[0].

    Equivalent to scanning one position at a time: a match consumes both ids,
    and a first id without its partner is copied before the search resumes.
    """
    first, second = pair
    output = []
    start = 0
    last = len(values) - 1
    while True:
        try:
            index = values.index(first, start, last)
        except ValueError:
            break
        if values[index + 1] == second:
            output.extend(values[start:index])
            output.append(replacement)
            start = index + 2
        else:
            output.extend(values[start : index + 1])
            start = index + 1
    output.extend(values[start:])
    return tuple(output)


def _weighted_pair_counts(streams):
    """Count pairs of trusted training streams, each weighted by multiplicity.

    Most corpus segments are unique, so weight-one streams are counted by
    Counter.update in C; repeated streams multiply their own pair counts.
    """
    counts = Counter()
    for stream, weight in streams.items():
        pairs = zip(stream, stream[1:])
        if weight == 1:
            counts.update(pairs)
        else:
            for pair, frequency in Counter(pairs).items():
                counts[pair] += frequency * weight
    return counts


def _merge_weighted_streams(streams, pair, replacement):
    """Apply one merge to trusted streams, adding weights of streams that converge."""
    merged = Counter()
    for stream, weight in streams.items():
        merged[_merge_known_pair(stream, pair, replacement)] += weight
    return merged


class ByteTokenizer:
    """Lossless UTF-8 byte BPE with immutable, dependency-ordered merge ranks."""

    def __init__(self, merges: tuple[tuple[int, int], ...] = ()):
        if not isinstance(merges, tuple) or len(merges) > 3840:
            raise TokenizerError("merges must be a tuple of at most 3840 pairs")
        vocabulary = [bytes([value]) for value in range(256)]
        storage_bytes = 256
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
            expansion = len(vocabulary[pair[0]]) + len(vocabulary[pair[1]])
            if expansion > 16384 or storage_bytes + expansion > 8 * 1024 * 1024:
                raise TokenizerError(
                    "expanded vocabulary exceeds its byte-storage budget"
                )
            storage_bytes += expansion
            vocabulary.append(vocabulary[pair[0]] + vocabulary[pair[1]])
        self._merges = merges
        self._vocabulary = tuple(vocabulary)
        self._ranks = {pair: index for index, pair in enumerate(merges)}

    @property
    def merges(self) -> tuple[tuple[int, int], ...]:
        """Merge pairs in rank order; the pair at index i created token 256 + i."""
        return self._merges

    @property
    def vocab_size(self) -> int:
        """256 byte tokens plus one token per merge, excluding special markers."""
        return len(self._vocabulary)

    @classmethod
    def train(cls, text: str, *, vocab_size: int = 320, min_frequency: int = 2):
        """Learn merges from text as one stream, so merges may cross its lines.

        Use tokenizer_training.train_documents to keep document boundaries.
        Stops early once no pair reaches min_frequency; ties go to the smallest
        pair. vocab_size outside [256, 4096], min_frequency below 2 or more than
        1 MiB of UTF-8 raise TokenizerError.
        """
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
        streams = Counter(tuple(segment.encode("utf-8")) for segment in segments)
        merges = []
        while len(merges) + 256 < vocab_size:
            counts = _weighted_pair_counts(streams)
            if not counts:
                break
            pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
            if counts[pair] < min_frequency:
                break
            streams = _merge_weighted_streams(streams, pair, 256 + len(merges))
            merges.append(pair)
        return cls(tuple(merges))

    def _encode_bytes(self, payload: bytes) -> tuple[int, ...]:
        """Apply the lowest-ranked present merge until no ranked pair remains.

        Byte values and merge ids are valid by construction, so each round skips
        the public helpers' revalidation without changing which merge applies.
        """
        values = tuple(payload)
        ranks = self._ranks
        while len(values) > 1:
            available = set(zip(values, values[1:])) & ranks.keys()
            if not available:
                break
            pair = min(available, key=ranks.__getitem__)
            values = _merge_known_pair(values, pair, 256 + ranks[pair])
        return values

    def encode_bytes(self, payload: bytes) -> tuple[int, ...]:
        """Encode arbitrary bytes without requiring them to be valid UTF-8."""
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        return self._encode_bytes(payload)

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode text losslessly; unseen characters fall back to byte tokens."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._encode_bytes(text.encode("utf-8"))

    def decode_bytes(self, values) -> bytes:
        """Join token bytes without UTF-8 decoding, for prefixes that end mid-character.

        Ids outside this vocabulary raise TokenizerError instead of indexing past it.
        """
        values = token_ids(values)
        if any(value >= self.vocab_size for value in values):
            raise TokenizerError("token id is outside the vocabulary")
        return b"".join(self._vocabulary[value] for value in values)

    def token_bytes(self, value: int) -> bytes:
        """Return the exact bytes one id expands to, for byte-level accounting.

        Serving masks, span offsets and training audits use this per-token view.
        """
        token_ids([value])
        if value >= self.vocab_size:
            raise TokenizerError("token id is outside the vocabulary")
        return self._vocabulary[value]

    def decode(self, values) -> str:
        """Decode strictly: ids cut mid-character raise instead of mangling text."""
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
        self._pattern_text = pattern

    @property
    def pattern(self):
        """The segmentation regex; checkpoints store it to keep piece boundaries."""
        return self._pattern_text

    def pieces(self, text):
        """Split text into regex pieces that must tile it exactly.

        A pattern that leaves a gap or matches empty text raises TokenizerError,
        because uncovered characters would silently vanish from the encoding.
        """
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
        """Learn merges inside regex pieces only, with ByteTokenizer.train's bounds.

        With the default pattern no token joins a word to adjacent punctuation or
        whitespace; repeated pieces are counted once, weighted by multiplicity.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
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
        """Encode each piece separately, so no token crosses a piece boundary."""
        return tuple(
            token
            for piece in self.pieces(text)
            for token in super(RegexTokenizer, self).encode(piece)
        )

    def encode_bytes(self, payload):
        """Always raise: raw bytes have no decoded text to split into pieces."""
        raise TokenizerError("regex tokenization requires decoded text boundaries")


class SpecialTokenizer:
    """Reserve named markers while rejecting them in untrusted ordinary input."""

    def __init__(self, tokenizer: ByteTokenizer, names: tuple[str, ...]):
        if not isinstance(tokenizer, ByteTokenizer):
            raise TypeError("tokenizer must be a byte tokenizer")
        if (
            not isinstance(names, tuple)
            or len(names) > 64
            or len(set(names)) != len(names)
            or any(
                not isinstance(name, str) or not 1 <= len(name) <= 128 for name in names
            )
        ):
            raise TokenizerError("special names must be unique bounded strings")
        try:
            encoded_names = tuple(name.encode("utf-8") for name in names)
        except UnicodeEncodeError as exc:
            raise TokenizerError("special names must be valid UTF-8 text") from exc
        self._tokenizer = tokenizer
        self._names = names
        self._special_ids = {
            name: tokenizer.vocab_size + index for index, name in enumerate(names)
        }
        self._special_bytes = encoded_names
        self._pattern = (
            re.compile(
                "|".join(
                    re.escape(name)
                    for name in sorted(names, key=lambda name: (-len(name), name))
                )
            )
            if names
            else None
        )

    @property
    def tokenizer(self):
        """The wrapped ordinary tokenizer, which never emits a reserved marker id."""
        return self._tokenizer

    @property
    def names(self):
        """Registered markers; names[i] has id tokenizer.vocab_size + i."""
        return self._names

    @property
    def vocab_size(self):
        """Ordinary vocabulary plus one reserved id per marker, for model sizing."""
        return self.tokenizer.vocab_size + len(self.names)

    def encode(self, text: str, *, allowed: frozenset[str] = frozenset()):
        """Encode text, mapping only markers named in allowed to their reserved ids.

        Any other registered marker in text raises TokenizerError, so untrusted
        input cannot inject control tokens; the longest marker wins an overlap.
        allowed must be a frozenset of registered names.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(allowed, frozenset) or not allowed.issubset(
            self._special_ids
        ):
            raise TokenizerError("allowed markers must name registered special tokens")
        if self._pattern is None:
            return self.tokenizer.encode(text)
        output = []
        position = 0
        for match in self._pattern.finditer(text):
            name = match.group()
            if name not in allowed:
                raise TokenizerError("special marker requires explicit authorization")
            output.extend(self.tokenizer.encode(text[position : match.start()]))
            output.append(self._special_ids[name])
            position = match.end()
        output.extend(self.tokenizer.encode(text[position:]))
        return tuple(output)

    def token_bytes(self, value):
        """Return an ordinary token's bytes or a reserved id's UTF-8 marker text."""
        token_ids([value])
        if value < self.tokenizer.vocab_size:
            return self.tokenizer.token_bytes(value)
        if value >= self.vocab_size:
            raise TokenizerError("token id is outside the vocabulary")
        return self._special_bytes[value - self.tokenizer.vocab_size]

    def decode_bytes(self, values):
        """Join token bytes, rendering each reserved id as its marker text."""
        return b"".join(self.token_bytes(value) for value in token_ids(values))

    def decode(self, values):
        """Decode strictly with markers as text, so authorized prompts round-trip."""
        return self.decode_bytes(values).decode("utf-8")
