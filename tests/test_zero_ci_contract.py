from pathlib import Path


def test_ci_runs_zero_contracts_and_retains_both_evidence_paths():
    repository = Path(__file__).resolve().parents[1]
    workflow = (repository / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    package = (repository / "pyproject.toml").read_text(encoding="utf-8")

    assert "tests/test_zero*.py" in workflow
    assert "ai-journey-zero-compare build" in workflow
    assert "ai-journey-zero-compare verify" in workflow
    assert "torch.distributed.run" in workflow
    assert "ai_journey.zero_probe" in workflow
    assert "ai-journey-zero-probe --verify-report" in workflow
    assert "zero-comparison.json" in workflow
    assert "cpu-gloo-probe.json" in workflow
    assert 'ai-journey-zero-compare = "ai_journey.zero_cli:main"' in package
    assert 'ai-journey-zero-probe = "ai_journey.zero_probe:main"' in package
