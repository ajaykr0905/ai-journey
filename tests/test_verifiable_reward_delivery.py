import json
from pathlib import Path
import tomllib


def test_command_is_packaged_and_installed_workflow_is_ci_gated():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["ai-journey-verifiable-rewards"] == (
        "ai_journey.verifiable_reward_cli:main"
    )
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    for phrase in (
        "python -m pytest -q tests/test_verifiable_reward_*.py",
        "ai-journey-verifiable-rewards build",
        "ai-journey-verifiable-rewards verify",
        "ai-journey-verifiable-rewards replay",
    ):
        assert phrase in workflow


def test_readiness_tracks_every_reward_source_test_and_operator_input():
    required = set(
        json.loads(Path("config/deployment_readiness.json").read_text())[
            "required_files"
        ]
    )
    expected = {
        str(path)
        for pattern in (
            "src/ai_journey/verifiable_reward_*.py",
            "tests/test_verifiable_reward_*.py",
        )
        for path in Path().glob(pattern)
    }
    expected.update(
        {"config/day-69-verifiable-rewards.json", "docs/verifiable-rewards.md"}
    )
    assert expected <= required


def test_operator_guide_preserves_claim_and_learner_boundaries():
    guide = Path("docs/verifiable-rewards.md").read_text(encoding="utf-8")
    for phrase in (
        "ai-journey-verifiable-rewards build",
        "does not train a policy",
        "false positives and false negatives",
        "Ajay still needs to personally watch",
    ):
        assert phrase in guide
    assert "| 69 | 2026-11-12 | Verifiable reward" in Path("README.md").read_text()
