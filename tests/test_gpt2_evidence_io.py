from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_journey.gpt2_audit import LogitParityResult
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_evidence import GPT2EvidenceError, build_evidence
from ai_journey.gpt2_evidence_io import load_evidence, write_evidence
from ai_journey.gpt2_source import GPT2Source


def _evidence():
    source = GPT2Source(
        "openai/gpt2", "6" * 40, "c.json", "a" * 64, "w.bin", "b" * 64, 1
    )
    config = GPT2Config(11, 4, 4, 2, 2, 1e-5, 0, 0, 0)
    parity = LogitParityResult((1, 2), 22, 1e-5, 1e-6, 2, 2, "c" * 64, "d" * 64)
    return build_evidence(
        source,
        config,
        state_fingerprint="e" * 64,
        parameter_count=312,
        parity=parity,
        torch_version="2.14.1",
        python_version="3.13.9",
    )


def test_atomically_round_trips_canonical_evidence(tmp_path: Path) -> None:
    destination = tmp_path / "evidence.json"

    write_evidence(destination, _evidence())

    assert load_evidence(destination) == _evidence()
    assert destination.read_bytes().endswith(b"\n")
    assert not list(tmp_path.glob("*.tmp"))


def test_replaces_an_existing_regular_evidence_file(tmp_path: Path) -> None:
    destination = tmp_path / "evidence.json"
    destination.write_text("old")

    write_evidence(destination, _evidence())

    assert load_evidence(destination).parameter_count == 312


def test_rejects_symlinked_destinations_and_sources(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(GPT2EvidenceError, match="symlink"):
        write_evidence(link, _evidence())
    with pytest.raises(GPT2EvidenceError, match="non-symlink"):
        load_evidence(link)


def test_rejects_duplicate_tampered_and_oversized_evidence(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}')
    tampered = tmp_path / "tampered.json"
    write_evidence(tampered, _evidence())
    payload = json.loads(tampered.read_text())
    payload["evidence"]["parameter_count"] += 1
    tampered.write_text(json.dumps(payload))

    with pytest.raises(GPT2EvidenceError, match="duplicate"):
        load_evidence(duplicate)
    with pytest.raises(GPT2EvidenceError, match="SHA-256 mismatch"):
        load_evidence(tampered)
    with pytest.raises(GPT2EvidenceError, match="byte limit"):
        load_evidence(tampered, max_bytes=10)
