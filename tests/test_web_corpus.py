from hashlib import sha256

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.web_corpus import ExtractionPolicy, extract_web_corpus
from ai_journey.web_source import WebPage


def _page(index, body=None, content_type="text/html; charset=utf-8"):
    text = (
        body
        or f"<article><h1>Page {index}</h1><p>{'useful public text ' * 6}</p></article>"
    )
    return WebPage(f"https://example.com/page-{index}", content_type, text.encode())


def test_corpus_extracts_one_hundred_pages_with_stable_unique_identities():
    pages = tuple(_page(index) for index in range(100))

    result = extract_web_corpus(pages)

    assert len(result.documents) == 100
    assert len({document.document_id for document in result.documents}) == 100
    assert result.documents[0].document_id == (
        "web-" + sha256(b"https://example.com/page-0").hexdigest()[:24]
    )
    assert result.rejected_pages == 0
    assert all(row.accepted for row in result.observations)


def test_corpus_records_short_non_html_and_encoding_quality_rejections():
    pages = (
        _page(0),
        _page(1, "<p>short</p>"),
        _page(2, "{}", "application/json"),
        WebPage(
            "https://example.com/page-3",
            "text/html; charset=ascii",
            b"<p>" + b"\xff" * 100 + b" useful text</p>",
        ),
    )
    policy = ExtractionPolicy(minimum_pages=4, minimum_text_characters=20)

    result = extract_web_corpus(pages, policy)

    assert len(result.documents) == 1
    assert result.rejected_pages == 3
    assert [row.rejection_reason for row in result.observations[1:]] == [
        "extracted text is shorter than the quality threshold",
        "web page content type must identify HTML",
        "decoded text exceeds the replacement-character threshold",
    ]


def test_corpus_enforces_page_count_and_unique_sources():
    policy = ExtractionPolicy(minimum_pages=2, minimum_text_characters=1)
    with pytest.raises(TokenizerError, match="minimum page count"):
        extract_web_corpus((_page(0),), policy)
    with pytest.raises(TokenizerError, match="unique"):
        extract_web_corpus((_page(0), _page(0)), policy)


def test_corpus_fails_if_every_page_is_rejected():
    policy = ExtractionPolicy(minimum_pages=1, minimum_text_characters=80)

    with pytest.raises(TokenizerError, match="accepted no documents"):
        extract_web_corpus((_page(0, "<p>short</p>"),), policy)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_pages": 0},
        {"minimum_text_characters": 0},
        {"maximum_replacement_fraction": 2.0},
        {"maximum_replacement_fraction": 1},
    ],
)
def test_extraction_policy_rejects_invalid_thresholds(kwargs):
    with pytest.raises(TokenizerError):
        ExtractionPolicy(**kwargs)
