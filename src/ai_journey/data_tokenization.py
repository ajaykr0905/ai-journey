"""Train-only BPE fitting and explicit corpus compression evidence."""

from dataclasses import asdict, dataclass

from .bpe import ByteTokenizer, TokenizerError
from .data_cleaning import CleanedDocument
from .tokenizer_io import tokenizer_fingerprint
from .tokenizer_metrics import CompressionMetrics, compression_metrics
from .tokenizer_training import train_documents


@dataclass(frozen=True)
class TokenizationReport:
    vocabulary_size: int
    tokenizer_sha256: str
    train: CompressionMetrics
    validation: CompressionMetrics
    total_tokens: int
    total_utf8_bytes: int
    bytes_per_token: float
    token_reduction_fraction: float

    def as_dict(self) -> dict:
        return asdict(self)


def build_tokenization_report(
    train: tuple[CleanedDocument, ...],
    validation: tuple[CleanedDocument, ...],
    *,
    vocabulary_size: int = 320,
    min_frequency: int = 2,
) -> tuple[ByteTokenizer, TokenizationReport]:
    """Fit on train only, then report exact content-token denominators."""
    for name, documents in (("train", train), ("validation", validation)):
        if (
            not isinstance(documents, tuple)
            or not documents
            or not all(isinstance(document, CleanedDocument) for document in documents)
        ):
            raise TokenizerError(f"{name} tokenization input is invalid")
    if {row.document_id for row in train} & {row.document_id for row in validation}:
        raise TokenizerError("tokenization partitions must be disjoint")
    tokenizer = train_documents(
        tuple(row.text for row in train),
        vocab_size=vocabulary_size,
        min_frequency=min_frequency,
    )
    train_metrics = compression_metrics(tokenizer, tuple(row.text for row in train))
    validation_metrics = compression_metrics(
        tokenizer, tuple(row.text for row in validation)
    )
    total_tokens = train_metrics.tokens + validation_metrics.tokens
    total_bytes = train_metrics.utf8_bytes + validation_metrics.utf8_bytes
    report = TokenizationReport(
        tokenizer.vocab_size,
        tokenizer_fingerprint(tokenizer),
        train_metrics,
        validation_metrics,
        total_tokens,
        total_bytes,
        total_bytes / total_tokens,
        1 - total_tokens / total_bytes,
    )
    return tokenizer, report
