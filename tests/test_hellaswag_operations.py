import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_reproduces_the_pinned_validation_slice():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for required in (
        "ruff check src/ai_journey/hellaswag*.py tests/test_hellaswag*.py",
        "datasets-server.huggingface.co/first-rows?dataset=Rowan%2Fhellaswag",
        "ai-journey-hellaswag evaluate",
        "--protocol config/hellaswag-validation.json",
        "ai-journey-hellaswag verify --report /tmp/hellaswag-evidence.json",
    ):
        assert required in workflow


def test_readiness_covers_the_complete_evaluation_surface():
    required_files = json.loads(
        (ROOT / "config" / "deployment_readiness.json").read_text(encoding="utf-8")
    )["required_files"]
    for path in (
        "src/ai_journey/hellaswag.py",
        "src/ai_journey/hellaswag_cli.py",
        "src/ai_journey/hellaswag_evidence.py",
        "config/hellaswag-validation.json",
        "docs/hellaswag-evaluation.md",
        "tests/test_hellaswag.py",
        "tests/test_hellaswag_cli.py",
        "tests/test_hellaswag_evidence.py",
        "tests/test_hellaswag_operations.py",
    ):
        assert path in required_files


def test_runbook_records_replay_commands_result_and_scope_limits():
    guide = (ROOT / "docs" / "hellaswag-evaluation.md").read_text(encoding="utf-8")
    for required in (
        "ai-journey-hellaswag evaluate",
        "ai-journey-hellaswag verify",
        "ai-journey-hellaswag compare",
        "completion-only",
        "25/100",
        "[0.17, 0.34]",
        "not an official 10,042-example",
        "proof of lecture viewing and learner understanding",
    ):
        assert required in guide


def test_readme_exposes_day_49_without_claiming_completion():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "selected verified code through Day 57" in readme
    assert "[bounded evaluation runbook](docs/hellaswag-evaluation.md)" in readme
    assert "| 49 | 2026-10-23 |" in readme
    assert "completed Day 49" not in readme
