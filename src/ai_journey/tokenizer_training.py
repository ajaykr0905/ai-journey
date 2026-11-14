"""Corpus-oriented BPE training without synthetic cross-document merges."""

from collections import Counter
from hashlib import sha256
import json

from .bpe import (
    ByteTokenizer,
    RegexTokenizer,
    TokenizerError,
    _weighted_pair_counts,
    _merge_weighted_streams,
)


def training_segments(documents: tuple[str, ...], *, pattern: str | None = None):
    """Check the corpus budget and return the streams that training keeps apart.

    Each document, or each regex piece of one, is its own stream, so no merge
    spans a boundary. An empty corpus, empty or non-text documents, more than
    100,000 documents or more than 1 MiB of UTF-8 raise TokenizerError.
    """
    if (
        not isinstance(documents, tuple)
        or not documents
        or any(not isinstance(text, str) or not text for text in documents)
    ):
        raise TokenizerError("training requires nonempty text documents")
    if (
        len(documents) > 100000
        or sum(len(text.encode("utf-8")) for text in documents) > 1024 * 1024
    ):
        raise TokenizerError("training documents exceed the bounded corpus budget")
    if pattern is None:
        return documents
    splitter = RegexTokenizer(pattern=pattern)
    return tuple(piece for document in documents for piece in splitter.pieces(document))


def train_documents(documents, *, vocab_size=320, min_frequency=2, pattern=None):
    """Treat each document and optional regex piece as an independent stream."""
    ByteTokenizer.train("", vocab_size=vocab_size, min_frequency=min_frequency)
    segments = training_segments(documents, pattern=pattern)
    trained = ByteTokenizer._train_segments(
        segments, vocab_size=vocab_size, min_frequency=min_frequency
    )
    return (
        trained if pattern is None else RegexTokenizer(trained.merges, pattern=pattern)
    )


class BPETrainingSession:
    """Interruptible deterministic training, bound to exact document boundaries."""

    def __init__(self, documents, *, min_frequency=2, pattern=None):
        ByteTokenizer.train("", min_frequency=min_frequency)
        segments = training_segments(documents, pattern=pattern)
        self._streams = Counter(tuple(piece.encode()) for piece in segments)
        self._identity = sha256(
            json.dumps(documents, ensure_ascii=False).encode()
        ).hexdigest()
        self._pattern = pattern
        self._min_frequency = min_frequency
        self._merges = []

    @property
    def tokenizer(self):
        """A byte or regex tokenizer, matching the session, for the merges so far."""
        merges = tuple(self._merges)
        return (
            ByteTokenizer(merges)
            if self._pattern is None
            else RegexTokenizer(merges, pattern=self._pattern)
        )

    def advance(self, merge_count):
        """Learn up to merge_count more merges and return the updated tokenizer.

        Stops early when pairs run out or fall below min_frequency; a count that
        would pass 3840 merges raises TokenizerError before any work.
        """
        if type(merge_count) is not int or not 0 <= merge_count <= 3840 - len(
            self._merges
        ):
            raise TokenizerError("merge budget exceeds the bounded vocabulary")
        for _ in range(merge_count):
            counts = _weighted_pair_counts(self._streams)
            if not counts:
                break
            pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
            if counts[pair] < self._min_frequency:
                break
            replacement = 256 + len(self._merges)
            self._streams = _merge_weighted_streams(self._streams, pair, replacement)
            self._merges.append(pair)
        return self.tokenizer

    def snapshot(self):
        """Checkpoint the corpus digest, training policy and merges, not pair counts.

        restore() recomputes the counts from the exact corpus instead.
        """
        return {
            "schema_version": 1,
            "corpus_sha256": self._identity,
            "pattern": self._pattern,
            "min_frequency": self._min_frequency,
            "merges": [list(pair) for pair in self._merges],
        }

    @classmethod
    def restore(cls, documents, state):
        """Rebuild a session by replaying a snapshot's merges on the same corpus.

        A wrong version or field set, another corpus digest, or merges that are
        not the canonical prefix training would choose raise TokenizerError, so
        an edited merge table cannot resume as if it had been learned.
        """
        if (
            not isinstance(state, dict)
            or set(state)
            != {"schema_version", "corpus_sha256", "pattern", "min_frequency", "merges"}
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
        ):
            raise TokenizerError("training checkpoint fields are invalid")
        merges = state["merges"]
        if (
            not isinstance(merges, list)
            or len(merges) > 3840
            or any(not isinstance(pair, list) or len(pair) != 2 for pair in merges)
        ):
            raise TokenizerError("training checkpoint merges are invalid")
        expected = ByteTokenizer(tuple(tuple(pair) for pair in merges)).merges
        session = cls(
            documents, min_frequency=state["min_frequency"], pattern=state["pattern"]
        )
        if session._identity != state["corpus_sha256"]:
            raise TokenizerError("training checkpoint corpus identity mismatch")
        # Verify the canonical prefix, not just a syntactically valid merge table.
        if session.advance(len(expected)).merges != expected:
            raise TokenizerError("training checkpoint is not a canonical merge prefix")
        return session
