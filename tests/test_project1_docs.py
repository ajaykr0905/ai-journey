from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_project1_guide_documents_complete_operator_workflow():
    guide = (ROOT / "docs/project1-run-operator.md").read_text()

    for command in (
        "ai-journey-project1 init",
        "ai-journey-project1 run",
        "ai-journey-project1 status",
        "ai-journey-project1 verify",
    ):
        assert command in guide
    assert "does not upload to Weights & Biases" in guide
    assert "not evidence that Ajay watched" in guide


def test_readme_links_to_project1_operator_guide():
    readme = (ROOT / "README.md").read_text()

    assert "[Project 1 run operator](docs/project1-run-operator.md)" in readme
