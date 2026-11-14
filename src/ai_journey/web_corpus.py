"""Batch HTML extraction with per-page quality evidence and stable documents."""

from dataclasses import dataclass
from hashlib import sha256

from .bpe import TokenizerError
from .data_documents import DataDocument
from .html_extraction import extract_visible_text
from .web_decode import decode_html
from .web_source import WebPage


@dataclass(frozen=True)
class ExtractionPolicy:
    """Page-count and per-page quality thresholds for one extraction run.

    config/html_extraction_100_pages.json pins 100 pages, 200 characters and a
    0.005 replacement fraction; the fraction must be a float in [0, 1].
    Invalid values raise TokenizerError.
    """

    minimum_pages: int = 100
    minimum_text_characters: int = 80
    maximum_replacement_fraction: float = 0.01

    def __post_init__(self) -> None:
        if type(self.minimum_pages) is not int or not 1 <= self.minimum_pages <= 1_000:
            raise TokenizerError("extraction minimum page count is invalid")
        if (
            type(self.minimum_text_characters) is not int
            or not 1 <= self.minimum_text_characters <= 1_000_000
        ):
            raise TokenizerError("extraction minimum text length is invalid")
        if (
            not isinstance(self.maximum_replacement_fraction, float)
            or not 0.0 <= self.maximum_replacement_fraction <= 1.0
        ):
            raise TokenizerError("extraction replacement threshold is invalid")


@dataclass(frozen=True)
class PageExtraction:
    """Outcome for every input page, so rejections stay in the denominator.

    extraction_ratio is extracted characters per payload byte; pages that fail
    to decode or parse record their reason with zero counts.
    """

    url: str
    payload_sha256: str
    payload_bytes: int
    encoding: str | None
    replacement_characters: int
    extracted_characters: int
    extracted_lines: int
    extraction_ratio: float
    accepted: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class WebCorpusResult:
    """Accepted documents with URL-derived ids and one observation per input page."""

    documents: tuple[DataDocument, ...]
    observations: tuple[PageExtraction, ...]

    @property
    def rejected_pages(self) -> int:
        """Pages observed and reported but kept out of the corpus."""
        return sum(not row.accepted for row in self.observations)


def _document_id(url: str) -> str:
    return f"web-{sha256(url.encode('utf-8')).hexdigest()[:24]}"


def _rejected(page: WebPage, reason: str) -> PageExtraction:
    return PageExtraction(
        page.url,
        sha256(page.payload).hexdigest(),
        len(page.payload),
        None,
        0,
        0,
        0,
        0.0,
        False,
        reason,
    )


def extract_web_corpus(
    pages: tuple[WebPage, ...], policy: ExtractionPolicy = ExtractionPolicy()
) -> WebCorpusResult:
    """Extract every page, retaining failures as inspectable observations."""
    if (
        not isinstance(pages, tuple)
        or not all(isinstance(page, WebPage) for page in pages)
        or not isinstance(policy, ExtractionPolicy)
    ):
        raise TypeError("web corpus extraction requires WebPage tuples and a policy")
    if len(pages) < policy.minimum_pages:
        raise TokenizerError("web corpus does not satisfy the minimum page count")
    if len({page.url for page in pages}) != len(pages):
        raise TokenizerError("web corpus page URLs must be unique")
    documents = []
    observations = []
    for page in pages:
        try:
            decoded = decode_html(page)
            extracted = extract_visible_text(decoded)
        except TokenizerError as exc:
            observations.append(_rejected(page, str(exc)))
            continue
        replacement_fraction = decoded.replacement_characters / max(
            len(decoded.html), 1
        )
        reason = None
        if extracted.text_characters < policy.minimum_text_characters:
            reason = "extracted text is shorter than the quality threshold"
        elif replacement_fraction > policy.maximum_replacement_fraction:
            reason = "decoded text exceeds the replacement-character threshold"
        accepted = reason is None
        observations.append(
            PageExtraction(
                page.url,
                sha256(page.payload).hexdigest(),
                len(page.payload),
                decoded.encoding,
                decoded.replacement_characters,
                extracted.text_characters,
                extracted.text_lines,
                extracted.text_characters / len(page.payload),
                accepted,
                reason,
            )
        )
        if accepted:
            documents.append(
                DataDocument(_document_id(page.url), extracted.text, page.url)
            )
    if not documents:
        raise TokenizerError("web corpus extraction accepted no documents")
    return WebCorpusResult(tuple(documents), tuple(observations))
