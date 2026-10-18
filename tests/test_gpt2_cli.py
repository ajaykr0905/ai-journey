from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ai_journey.gpt2_audit import LogitParityResult
from ai_journey.gpt2_cli import GPT2CLIError, load_reference_probe, main, verify_report
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_evidence import build_evidence
from ai_journey.gpt2_evidence_io import write_evidence
from ai_journey.gpt2_source import GPT2Source


def _report(path: Path, *, max_error: float = 1e-5, top1_matches: int = 2) -> None:
    source = GPT2Source(
        "openai/gpt2", "6" * 40, "c.json", "a" * 64, "w.bin", "b" * 64, 1
    )
    config = GPT2Config(11, 4, 4, 2, 2, 1e-5, 0, 0, 0)
    parity = LogitParityResult(
        (1, 2), 22, max_error, 1e-6, top1_matches, 2, "c" * 64, "d" * 64
    )
    evidence = build_evidence(
        source,
        config,
        state_fingerprint="e" * 64,
        parameter_count=312,
        parity=parity,
        torch_version="2.14.1",
        python_version="3.13.9",
    )
    write_evidence(path, evidence)


def test_loads_only_bounded_exact_reference_tensor_archives(tmp_path: Path) -> None:
    reference = tmp_path / "reference.pt"
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    logits = torch.zeros(1, 2, 11)
    torch.save({"token_ids": tokens, "logits": logits}, reference)

    loaded_tokens, loaded_logits = load_reference_probe(reference)

    torch.testing.assert_close(loaded_tokens, tokens)
    torch.testing.assert_close(loaded_logits, logits)


def test_rejects_reference_probe_with_unknown_fields_or_symlink(tmp_path: Path) -> None:
    reference = tmp_path / "reference.pt"
    torch.save(
        {"token_ids": torch.ones(1), "logits": torch.ones(1), "extra": 1}, reference
    )
    link = tmp_path / "link.pt"
    link.symlink_to(reference)

    with pytest.raises(GPT2CLIError, match="only token_ids and logits"):
        load_reference_probe(reference)
    with pytest.raises(GPT2CLIError, match="non-symlink"):
        load_reference_probe(link)


def test_verify_command_enforces_integrity_error_and_top1_gates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    passing = tmp_path / "passing.json"
    excessive = tmp_path / "excessive.json"
    incomplete = tmp_path / "incomplete.json"
    _report(passing)
    _report(excessive, max_error=0.1)
    _report(incomplete, top1_matches=1)

    assert main(["verify", "--report", str(passing)]) == 0
    assert "evidence_sha256" in capsys.readouterr().out
    with pytest.raises(GPT2CLIError, match="exceeds"):
        verify_report(excessive, max_absolute_error=1e-3)
    with pytest.raises(GPT2CLIError, match="top-1"):
        verify_report(incomplete, max_absolute_error=1e-3)
