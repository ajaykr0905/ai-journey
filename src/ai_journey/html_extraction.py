"""Deterministic visible-text extraction from decoded HTML pages."""

from dataclasses import dataclass
from html.parser import HTMLParser

from .bpe import TokenizerError
from .web_decode import DecodedPage


_BLOCK_TAGS = {
    "article",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
_SUPPRESSED_TAGS = {
    "canvas",
    "footer",
    "form",
    "head",
    "header",
    "nav",
    "noscript",
    "script",
    "style",
    "svg",
    "template",
}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


@dataclass(frozen=True)
class ExtractedPage:
    """Visible text as normalized lines, plus the page title collected on its own.

    text_characters and text_lines measure the normalized text that extraction
    policies threshold.
    """

    url: str
    title: str | None
    text: str
    text_characters: int
    text_lines: int


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.stack: list[tuple[str, bool]] = []
        self.suppressed_depth = 0
        self.title_depth = 0

    def handle_starttag(self, tag, attrs) -> None:
        tag = tag.lower()
        suppressed = tag in _SUPPRESSED_TAGS
        if suppressed:
            self.suppressed_depth += 1
        if tag == "title":
            self.title_depth += 1
        if tag not in _VOID_TAGS:
            self.stack.append((tag, suppressed))
        if not self.suppressed_depth and tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if not self.suppressed_depth and tag == "img":
            alt = next((value for name, value in attrs if name.lower() == "alt"), None)
            if alt:
                self.parts.append(alt)

    def handle_startendtag(self, tag, attrs) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag) -> None:
        tag = tag.lower()
        if tag == "title" and self.title_depth:
            self.title_depth -= 1
        matching = next(
            (
                index
                for index in range(len(self.stack) - 1, -1, -1)
                if self.stack[index][0] == tag
            ),
            None,
        )
        if matching is not None:
            removed = self.stack[matching:]
            del self.stack[matching:]
            self.suppressed_depth -= sum(suppressed for _, suppressed in removed)
        if not self.suppressed_depth and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data) -> None:
        if self.title_depth:
            self.title_parts.append(data)
        if not self.suppressed_depth:
            self.parts.append(data)


def _normalized_lines(parts: list[str]) -> tuple[str, ...]:
    joined = "".join(parts).replace("\r\n", "\n").replace("\r", "\n")
    return tuple(line for raw in joined.split("\n") if (line := " ".join(raw.split())))


def extract_visible_text(page: DecodedPage) -> ExtractedPage:
    """Remove executable and navigation markup while preserving text blocks."""
    if not isinstance(page, DecodedPage):
        raise TypeError("HTML extraction requires a DecodedPage")
    parser = _VisibleTextParser()
    try:
        parser.feed(page.html)
        parser.close()
    except (AssertionError, ValueError) as exc:
        raise TokenizerError("unable to parse HTML payload") from exc
    lines = _normalized_lines(parser.parts)
    if not lines:
        raise TokenizerError("HTML extraction produced no visible text")
    title = " ".join(" ".join(parser.title_parts).split()) or None
    text = "\n".join(lines)
    return ExtractedPage(page.url, title, text, len(text), len(lines))
