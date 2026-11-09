from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_gates_minhash_source_format_tests_and_installed_replay():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "Check replayable MinHash deduplication contracts" in workflow
    assert "ruff check src/ai_journey/minhash_*.py tests/test_minhash_*.py" in workflow
    assert "python -m pytest -q tests/test_minhash_*.py" in workflow
    assert "ai-journey-minhash build" in workflow
    assert "ai-journey-minhash verify" in workflow
    assert "ai-journey-minhash replay" in workflow
    assert "${{ runner.temp }}/minhash-evidence/" in workflow
