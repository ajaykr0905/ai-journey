import json

import pytest

from ai_journey.gpt2_training import GPT2TrainingError
from ai_journey.gpt2_training_evidence import build_training_evidence
from ai_journey.gpt2_training_evidence_io import (
    load_training_evidence,
    write_training_evidence,
)
from test_gpt2_training_evidence import run_fixture


def test_training_evidence_round_trips_atomically(tmp_path):
    payload = build_training_evidence(run_fixture(tmp_path))
    path = tmp_path / "evidence.json"

    write_training_evidence(path, payload)

    assert load_training_evidence(path) == payload
    assert path.read_bytes().endswith(b"\n")
    assert not list(tmp_path.glob(".evidence.json.*.tmp"))


def test_training_evidence_loader_enforces_byte_budget(tmp_path):
    payload = build_training_evidence(run_fixture(tmp_path))
    path = tmp_path / "evidence.json"
    write_training_evidence(path, payload)

    with pytest.raises(GPT2TrainingError, match="byte budget"):
        load_training_evidence(path, max_bytes=path.stat().st_size - 1)


def test_training_evidence_loader_rejects_symlinks(tmp_path):
    payload = build_training_evidence(run_fixture(tmp_path))
    target = tmp_path / "target.json"
    write_training_evidence(target, payload)
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(GPT2TrainingError, match="non-symlink"):
        load_training_evidence(link)


def test_training_evidence_writer_rejects_unverified_payload(tmp_path):
    payload = build_training_evidence(run_fixture(tmp_path))
    payload["parameter_count"] += 1

    with pytest.raises(GPT2TrainingError, match="digest"):
        write_training_evidence(tmp_path / "evidence.json", payload)


def test_training_evidence_loader_rejects_non_json(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"not": "evidence"}), encoding="utf-8")

    with pytest.raises(GPT2TrainingError, match="fields"):
        load_training_evidence(path)
