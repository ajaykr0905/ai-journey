import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.web_decode import MAX_HTML_BYTES, decode_html
from ai_journey.web_source import WebPage


def _page(payload, content_type="text/html"):
    return WebPage("https://example.com/page", content_type, payload)


def test_decoder_prefers_http_charset_and_records_replacements():
    decoded = decode_html(_page(b"caf\xe9\xff", "text/html; charset=ascii"))

    assert decoded.encoding == "ascii"
    assert decoded.html == "caf\ufffd\ufffd"
    assert decoded.replacement_characters == 2


def test_decoder_sniffs_bounded_meta_charset_before_utf8_default():
    payload = b'<meta charset="windows-1252"><p>price \x80</p>'

    decoded = decode_html(_page(payload))

    assert decoded.encoding == "cp1252"
    assert "price €" in decoded.html


def test_decoder_defaults_to_utf8_for_undeclared_html():
    decoded = decode_html(_page("<p>नमस्ते</p>".encode()))

    assert decoded.encoding == "utf-8"
    assert decoded.replacement_characters == 0


@pytest.mark.parametrize(
    ("content_type", "message"),
    [
        ("application/json", "identify HTML"),
        ("text/html; charset=utf-8; charset=ascii", "repeats"),
        ("text/html; charset=not-real", "unsupported"),
    ],
)
def test_decoder_rejects_ambiguous_or_non_html_metadata(content_type, message):
    with pytest.raises(TokenizerError, match=message):
        decode_html(_page(b"<p>x</p>", content_type))


def test_decoder_enforces_configurable_and_absolute_byte_budgets():
    with pytest.raises(TokenizerError, match="byte budget"):
        decode_html(_page(b"1234"), max_bytes=3)
    with pytest.raises(TokenizerError, match="byte budget"):
        decode_html(_page(b"x"), max_bytes=MAX_HTML_BYTES + 1)
