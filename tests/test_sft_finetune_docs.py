from pathlib import Path


def test_operator_guide_documents_commands_limits_and_learner_evidence():
    guide = Path("docs/sft-finetuning.md").read_text(encoding="utf-8")
    for phrase in (
        "ai-journey-sft-finetune build",
        "ai-journey-sft-finetune verify",
        "ai-journey-sft-finetune replay",
        "not a useful aligned model",
        "Ajay still needs to personally watch",
    ):
        assert phrase in guide
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "| 68 | 2026-11-11 | Public-data assistant-only SFT" in readme
