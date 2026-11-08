import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.html_extraction import extract_visible_text
from ai_journey.web_decode import DecodedPage


def _decoded(html):
    return DecodedPage("https://example.com/page", html, "utf-8", 0)


def test_extractor_keeps_visible_blocks_title_entities_and_image_alt_text():
    page = _decoded(
        """
        <html><head><title> Useful &amp; Public </title></head>
        <body><main><h1>Heading</h1><p>First   paragraph.</p>
        <img alt="informative chart"><p>Second &lt; paragraph.</p></main></body></html>
        """
    )

    extracted = extract_visible_text(page)

    assert extracted.title == "Useful & Public"
    assert extracted.text == (
        "Heading\nFirst paragraph.\ninformative chart\nSecond < paragraph."
    )
    assert extracted.text_lines == 4
    assert extracted.text_characters == len(extracted.text)


def test_extractor_removes_code_navigation_forms_and_page_chrome():
    page = _decoded(
        """
        <header>brand</header><nav>menu</nav><article>Keep me.
        <script>steal()</script><style>.hidden{}</style><form>email</form>
        <noscript>fallback chrome</noscript></article><footer>copyright</footer>
        """
    )

    assert extract_visible_text(page).text == "Keep me."


def test_extractor_recovers_after_malformed_suppressed_markup():
    page = _decoded("<nav><div>menu</nav><article><p>Useful text")

    assert extract_visible_text(page).text == "Useful text"


def test_extractor_rejects_markup_without_visible_content():
    with pytest.raises(TokenizerError, match="no visible text"):
        extract_visible_text(_decoded("<script>only_code()</script>"))
