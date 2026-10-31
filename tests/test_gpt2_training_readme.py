from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_exposes_training_controls_without_claiming_learner_completion():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "[training controls guide](docs/gpt2-training-controls.md)" in readme
    assert "ai-journey-gpt2-train verify" in readme
    assert "ai-journey-gpt2-train replay" in readme
    assert "selected verified code through Day 57" in readme
    assert "completed Day 47" not in readme
