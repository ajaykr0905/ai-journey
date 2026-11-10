from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repository_surfaces_day62_without_claiming_execution_or_learning():
    readme = (ROOT / "README.md").read_text()
    progress = (ROOT / "PROGRESS.md").read_text()
    guide = (ROOT / "docs/scaling-experiment-protocol.md").read_text()

    assert "selected verified code through Day 67" in readme
    assert "[Day 62 scaling experiment guide]" in readme
    assert "| 62 | 2026-11-05 |" in readme
    assert "| 62 | Complete model/token/seed grid" in progress
    assert "Stanford CS336 Lecture 11" in progress
    assert "They are not training measurements" in guide
    assert "The residual band is not a" in guide
    assert "confidence interval" in guide
    assert "lecture-viewing evidence" in guide


def test_guide_documents_installed_build_and_verify_commands():
    guide = (ROOT / "docs/scaling-experiment-protocol.md").read_text()

    assert "ai-journey-scaling-experiment build" in guide
    assert "ai-journey-scaling-experiment verify" in guide
    assert "config/day-62-scaling-experiment-fixture.json" in guide
