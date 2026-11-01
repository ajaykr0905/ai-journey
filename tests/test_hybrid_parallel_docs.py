from pathlib import Path


def test_runbook_documents_exact_commands_and_evidence_boundary():
    runbook = (
        Path(__file__).resolve().parents[1] / "docs" / "hybrid-parallelism.md"
    ).read_text(encoding="utf-8")

    for required in (
        "ai-journey-hybrid-parallel",
        "ai-journey-hybrid-parallel-verify",
        "--world-size 8",
        "--tensor-shape 2,3",
        "--activation-shape 4",
        "--payload-hex 000102030405060708090a0b0c0d0e0f",
        "CPU/Gloo",
        "No NVIDIA GPU or NCCL",
        "not performance, throughput, utilization, memory, cost, multi-node, or production-readiness evidence",
        "not evidence that Ajay watched a lecture or completed a manual exercise",
    ):
        assert required in runbook
