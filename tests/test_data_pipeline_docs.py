from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_operator_guide_contains_reproducible_commands_and_exact_observations():
    guide = (ROOT / "docs/day-59-data-pipeline.md").read_text()

    for command in (
        "ai-journey-data-pipeline build",
        "ai-journey-data-pipeline verify",
        "ai-journey-data-pipeline accept",
    ):
        assert command in guide
    assert "69,626 content tokens" in guide
    assert "110,379 UTF-8 bytes" in guide
    assert "1.5853129578031195 bytes per token" in guide
    assert "CC-BY-SA-3.0" in guide


def test_operator_guide_states_atomic_recovery_and_claim_boundaries():
    guide = (ROOT / "docs/day-59-data-pipeline.md").read_text()

    assert "writes the manifest last" in guide
    assert "never delete or overwrite" in guide
    assert "no GPU" in guide
    assert "supporting quantization video was watched" in guide


def test_repository_entry_points_expose_day59_without_claiming_learning_completion():
    readme = (ROOT / "README.md").read_text()
    progress = (ROOT / "PROGRESS.md").read_text()

    assert "selected verified code through Day 59" in readme
    assert "[Day 59 data pipeline guide](docs/day-59-data-pipeline.md)" in readme
    assert "| 59 | Pinned English PUD ingestion" in progress
    assert "quantization support video pending user confirmation" in progress
