from pathlib import Path


def test_zero_runbook_documents_replay_and_honest_evidence_boundaries():
    repository = Path(__file__).resolve().parents[1]
    runbook_path = repository / "docs" / "zero-stage-comparison.md"
    readme = (repository / "README.md").read_text(encoding="utf-8")

    assert runbook_path.exists()
    runbook = runbook_path.read_text(encoding="utf-8")
    assert "ai-journey-zero-compare build" in runbook
    assert "ai-journey-zero-compare verify" in runbook
    assert "torch.distributed.run" in runbook
    assert "ai-journey-zero-probe --verify-report" in runbook
    assert "CPU/Gloo" in runbook
    assert "no NVIDIA GPU" in runbook
    assert "does not satisfy the Week 9 real multi-GPU gate" in runbook
    assert "docs/zero-stage-comparison.md" in readme
