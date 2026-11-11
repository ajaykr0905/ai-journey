from pathlib import Path
import tomllib


def test_finetune_command_is_packaged_and_public_data_is_ci_gated():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["ai-journey-sft-finetune"] == (
        "ai_journey.sft_finetune_cli:main"
    )
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python -m pytest -q tests/test_sft_finetune_*.py" in workflow
    assert "ai-journey-sft-finetune build" in workflow
    assert "ai-journey-sft-finetune verify" in workflow
    assert "ai-journey-sft-finetune replay" in workflow
