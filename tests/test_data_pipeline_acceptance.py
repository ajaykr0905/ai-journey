import json
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.data_pipeline_acceptance import (
    load_acceptance_contract,
    verify_pipeline_acceptance,
)


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_acceptance_contract_records_public_day59_observations():
    contract = load_acceptance_contract(
        ROOT / "config/day-59-data-pipeline-acceptance.json"
    )

    assert contract["source_sha256"].startswith("63eb7e")
    assert contract["input_documents"] == 1000
    assert contract["quality_rejected"] == 3
    assert contract["train_documents"] == 897
    assert contract["validation_documents"] == 100
    assert contract["total_tokens"] == 69626


def test_acceptance_compares_replayed_evidence_with_exact_inputs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    config = tmp_path / "config"
    contract_path = tmp_path / "acceptance.json"
    source.write_bytes(b"source")
    config.write_bytes(b"config")
    report = {
        "report_sha256": "a" * 64,
        "counts": {
            "input_documents": 10,
            "quality_rejected": 1,
            "exact_duplicates": 2,
        },
        "partitions": {"train": {"documents": 6}, "validation": {"documents": 1}},
    }
    replay = {
        "tokenizer_sha256": "b" * 64,
        "total_tokens": 50,
        "total_utf8_bytes": 80,
    }
    monkeypatch.setattr(
        "ai_journey.data_pipeline_acceptance.verify_pipeline_artifacts",
        lambda _path: report,
    )
    monkeypatch.setattr(
        "ai_journey.data_pipeline_acceptance.replay_pipeline_artifacts",
        lambda _path: replay,
    )
    contract = {
        "schema_version": 1,
        "source_sha256": sha256(b"source").hexdigest(),
        "config_sha256": sha256(b"config").hexdigest(),
        "report_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "input_documents": 10,
        "quality_rejected": 1,
        "exact_duplicates": 2,
        "train_documents": 6,
        "validation_documents": 1,
        "total_tokens": 50,
        "total_utf8_bytes": 80,
    }
    contract_path.write_text(json.dumps(contract))

    assert (
        verify_pipeline_acceptance(tmp_path / "dataset", contract_path, config, source)
        == contract
    )
    contract["total_tokens"] += 1
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(TokenizerError, match="acceptance contract"):
        verify_pipeline_acceptance(tmp_path / "dataset", contract_path, config, source)
