"""Corpus-oriented BPE training without synthetic cross-document merges."""

from .bpe import ByteTokenizer, RegexTokenizer, TokenizerError


def training_segments(documents: tuple[str, ...], *, pattern: str | None = None):
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
