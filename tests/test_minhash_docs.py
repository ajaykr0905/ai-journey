from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_links_the_day66_minhash_operator_guide():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "selected verified code through Day 67" in readme
    assert "[MinHash deduplication guide](docs/minhash-deduplication.md)" in readme


def test_guide_documents_build_verify_replay_metrics_and_boundaries():
    guide = (ROOT / "docs" / "minhash-deduplication.md").read_text(encoding="utf-8")

    assert "ai-journey-minhash build" in guide
    assert "ai-journey-minhash verify" in guide
    assert "ai-journey-minhash replay" in guide
    assert "removed_fraction" in guide
    assert "exact Jaccard" in guide
    assert "do not prove that Ajay ran the command" in guide


def test_progress_keeps_learner_evidence_explicitly_pending():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")

    assert "| 66 |" in progress
    assert "Learner-run deduplication over the Project 1 corpus" in progress
    assert "Stanford CS336 Lecture 14" in progress
