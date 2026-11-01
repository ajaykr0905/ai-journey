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
        "does not independently authenticate that execution occurred",
        "No NVIDIA GPU or NCCL",
        "not performance, throughput, utilization, memory, cost, multi-node, or production-readiness evidence",
        "not evidence that Ajay watched a lecture or completed a manual exercise",
    ):
        assert required in runbook


def test_program_ledger_records_every_source_commit_and_remaining_shortfall():
    ledger = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "ai-infrastructure-program-ledger.md"
    ).read_text(encoding="utf-8")

    for source_sha in (
        "b10f527a1d33996113bff7e6f9340412eade2acc",
        "d8dff3b18dc978ac4ec83a7737ea242929ea14e7",
        "c065a22531b6a7296cd7cb4cfbda689c980a9e80",
        "6f2bf3a4c80eae65eeadd790d4fff6647662efa1",
        "ca1ee2006baafbc6ed53430cb9e9079101d9c222",
        "11b9b9a48249b25058f19ddded91bd3823385afd",
        "d0bec000bcc181203bd93bc4c3b715347fd12fc0",
        "edfa5b8bb5e592b2ee1c2847b01c7f6a891bfef2",
        "987ad71a57207fa9060766f18a82f181dab1af64",
        "1187a7267461e455435f4d5c6fdbe097bf51a367",
        "22291d888cf920774efa1ee3eb5c74372e64df25",
        "6ac850dcc48709d7161fb2cf54650cbcd548ff39",
        "844f39289542b76c6c720250156bf131e751a249",
        "8dcebd463f9d619a9b1939735fe4d9f589beb173",
    ):
        assert source_sha in ledger
    assert "14 / 70" in ledger
    assert "56 meaningful source commits" in ledger
    assert "ledger-only bookkeeping commits are excluded" in ledger
    assert "Update hybrid parallelism program ledger" in ledger
    assert "does not independently authenticate execution" in ledger
