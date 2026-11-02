from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_builds_replays_and_accepts_the_installed_public_pipeline():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "data-pipeline-cpu:" in workflow
    assert "python -m pytest -q tests/test_data_*.py" in workflow
    assert "python -m build --wheel --outdir" in workflow
    assert "ai-journey-data-pipeline build" in workflow
    assert "ai-journey-data-pipeline verify" in workflow
    assert "ai-journey-data-pipeline accept" in workflow
    assert "6e328e8195c04a297c1e375d21fe663be1ce6178" in workflow


def test_package_exposes_the_data_pipeline_operator_command():
    project = (ROOT / "pyproject.toml").read_text()

    assert 'ai-journey-data-pipeline = "ai_journey.data_pipeline_cli:main"' in project
