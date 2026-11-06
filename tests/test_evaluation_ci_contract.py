from pathlib import Path


def test_ci_runs_contracts_pinned_benchmarks_and_suite_verification():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text()
    for text in (
        "Check three-axis evaluation contracts",
        "tests/test_evaluation*.py tests/test_lambada*.py",
        "config/lambada-test.json",
        "ai-journey-lambada verify --report /tmp/lambada-evidence.json",
        "ai-journey-evaluation-suite build",
        "ai-journey-evaluation-suite verify --report /tmp/evaluation-suite.json",
    ):
        assert text in workflow
