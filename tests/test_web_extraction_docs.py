from pathlib import Path


def test_html_extraction_guide_documents_operating_and_evidence_boundaries():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    guide = (root / "docs/html-extraction.md").read_text(encoding="utf-8")

    assert "[HTML extraction guide](docs/html-extraction.md)" in readme
    assert "ai-journey-html-extract build" in guide
    assert "ai-journey-html-extract verify" in guide
    assert "config/html_extraction_100_pages.json" in guide
    assert "claim that Ajay selected or inspected 100 real pages" in guide
    assert "robots policies, terms, rate limits, and source licenses" in guide
