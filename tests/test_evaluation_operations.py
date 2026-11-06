from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_guide_documents_reproducible_commands_and_claim_limits():
    guide = (ROOT / "docs/evaluation-harness.md").read_text()
    for text in (
        "ai-journey-lambada evaluate",
        "ai-journey-lambada verify",
        "ai-journey-lambada compare",
        "ai-journey-evaluation-suite build",
        "ai-journey-evaluation-suite verify",
        "b304b6afffcc2d23ba8b6a67d16df47c60f7511f8ea2ef60016d331b3a485e15",
        "not evidence of general intelligence",
        "lecture viewing",
    ):
        assert text in guide


def test_readme_links_day_63_evaluation_harness():
    readme = (ROOT / "README.md").read_text()
    assert "selected verified code through Day 63" in readme
    assert "[evaluation harness guide](docs/evaluation-harness.md)" in readme
    assert "| 63 | 2026-11-06 |" in readme
