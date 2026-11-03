from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scaling_guide_contains_reproducible_commands_and_exact_results():
    guide = (ROOT / "docs/scaling-law-planning.md").read_text()

    assert "ai-journey-scaling build" in guide
    assert "ai-journey-scaling verify" in guide
    assert "2,000,000,000 tokens" in guide
    assert "15,259 steps" in guide
    assert "27,648-token overshoot" in guide
    assert "28,724 complete passes" in guide
    assert "62,776 tokens" in guide


def test_scaling_guide_states_analytical_and_learner_boundaries():
    guide = (ROOT / "docs/scaling-law-planning.md").read_text()

    assert "not an empirical loss curve" in guide
    assert "No GPU" in guide
    assert "Ajay must separately confirm" in guide
    assert "invalid input leaves an existing" in guide
