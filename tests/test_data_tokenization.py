import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_cleaning import clean_documents
from ai_journey.data_documents import DataDocument
from ai_journey.data_tokenization import build_tokenization_report


def _documents(prefix, texts):
    return clean_documents(
        tuple(
            DataDocument(f"{prefix}-{index}", text, "public://snapshot")
            for index, text in enumerate(texts)
        )
    )


def test_tokenization_reports_exact_tokens_bytes_and_compression_ratio():
    train = _documents("train", ("alpha alpha alpha", "beta beta beta"))
    validation = _documents("validation", ("alpha beta",))

    tokenizer, report = build_tokenization_report(
        train, validation, vocabulary_size=300
    )

    assert report.vocabulary_size == tokenizer.vocab_size
    assert len(report.tokenizer_sha256) == 64
    assert report.total_tokens == report.train.tokens + report.validation.tokens
    assert report.total_utf8_bytes == sum(
        len(row.text.encode()) for row in train + validation
    )
    assert report.bytes_per_token == report.total_utf8_bytes / report.total_tokens
    assert report.token_reduction_fraction == 1 - (
        report.total_tokens / report.total_utf8_bytes
    )
    assert all(
        tokenizer.decode(tokenizer.encode(row.text)) == row.text for row in train
    )


def test_tokenizer_is_fit_only_on_training_documents():
    train = _documents("train", ("abababab",))
    validation = _documents("validation", ("zzzzzzzz",))

    tokenizer, _ = build_tokenization_report(train, validation, vocabulary_size=260)

    assert tokenizer.encode("abababab") != tuple("abababab".encode())
    assert tokenizer.encode("zzzzzzzz") == tuple("zzzzzzzz".encode())


def test_tokenization_rejects_partition_identity_overlap():
    train = _documents("same", ("one text",))
    validation = _documents("same", ("other text",))

    with pytest.raises(TokenizerError, match="disjoint"):
        build_tokenization_report(train, validation)
