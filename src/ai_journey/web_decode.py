"""Bounded, explicit HTML byte decoding for reproducible corpus extraction."""

import codecs
from dataclasses import dataclass
import re

from .bpe import TokenizerError
from .web_source import WebPage


MAX_HTML_BYTES = 2 * 1024 * 1024
_META_CHARSET = re.compile(
    rb"<meta\s+[^>]*charset\s*=\s*[\"']?\s*([a-zA-Z0-9._:-]+)", re.IGNORECASE
)


@dataclass(frozen=True)
class DecodedPage:
    """Decoded HTML, the canonical codec chosen, and its U+FFFD count.

    Decoding replaces bad bytes instead of failing, so the policy can reject a
    lossy page by its replacement fraction rather than lose it.
    """

    url: str
    html: str
    encoding: str
    replacement_characters: int


def _declared_charset(content_type: str) -> str | None:
    parts = [part.strip() for part in content_type.split(";")]
    if parts[0].lower() not in {"text/html", "application/xhtml+xml"}:
        raise TokenizerError("web page content type must identify HTML")
    values = []
    for parameter in parts[1:]:
        name, separator, value = parameter.partition("=")
        if separator and name.strip().lower() == "charset":
            values.append(value.strip().strip("\"'"))
    if len(values) > 1:
        raise TokenizerError("web page content type repeats the charset")
    return values[0] if values else None


def decode_html(page: WebPage, *, max_bytes: int = MAX_HTML_BYTES) -> DecodedPage:
    """Decode one HTML payload with a recorded, deterministic encoding choice."""
    if not isinstance(page, WebPage):
        raise TypeError("HTML decoding requires a WebPage")
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_HTML_BYTES:
        raise TokenizerError("HTML byte budget is invalid")
    if len(page.payload) > max_bytes:
        raise TokenizerError("web page payload exceeds the HTML byte budget")
    encoding = _declared_charset(page.content_type)
    if encoding is None:
        match = _META_CHARSET.search(page.payload[:1024])
        encoding = match.group(1).decode("ascii") if match else "utf-8"
    try:
        canonical_encoding = codecs.lookup(encoding).name
    except LookupError as exc:
        raise TokenizerError("web page declares an unsupported charset") from exc
    html = page.payload.decode(canonical_encoding, errors="replace")
    return DecodedPage(
        page.url,
        html,
        canonical_encoding,
        html.count("\ufffd"),
    )
