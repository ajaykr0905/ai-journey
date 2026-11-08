import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.web_source import WebPage, canonical_public_url


def test_public_url_canonicalizes_host_default_port_path_and_fragment():
    assert (
        canonical_public_url("https://EXAMPLE.com:443?chapter=1#heading")
        == "https://example.com/?chapter=1"
    )


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/page.html",
        "http://localhost/page",
        "http://127.0.0.1/page",
        "http://10.0.0.1/page",
        "https://user:secret@example.com/page",
        "https://example.com:8443/page",
        "https://intranet/page",
    ],
)
def test_public_url_rejects_nonpublic_or_ambiguous_sources(url):
    with pytest.raises(TokenizerError):
        canonical_public_url(url)


def test_web_page_requires_canonical_identity_and_immutable_bytes():
    page = WebPage("https://example.com/page", "text/html; charset=utf-8", b"<p>x</p>")

    assert page.payload == b"<p>x</p>"
    with pytest.raises(TokenizerError, match="canonical"):
        WebPage("https://EXAMPLE.com/page", "text/html", b"<p>x</p>")
