from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_runs_training_integrity_and_exact_replay_gates():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    for command in (
        "ruff check src/ai_journey/gpt2*.py tests/test_gpt2*.py",
        "python -m pytest -q tests/test_gpt2*.py",
        "ai-journey-gpt2-train verify --report artifacts/gpt2-training-2026-10-21/training-evidence.json",
        "ai-journey-gpt2-train train --source data/day-19-demo-names.txt --spec config/gpt2-training.json --checkpoint /tmp/gpt2-training.pt --report /tmp/gpt2-training.json",
        "ai-journey-gpt2-train replay --source data/day-19-demo-names.txt --report /tmp/gpt2-training.json",
    ):
        assert command in workflow
