from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import pytest

from ai_journey.sft_dataset_build import SFTDatasetPolicy
from ai_journey.sft_dataset_cli import load_build_config, main
from ai_journey.sft_dataset_source import SFTSourceSpec
from ai_journey.sft_dataset_verify import verify_sft_dataset


def snapshot() -> bytes:
    categories = ("open_qa", "classification", "summarization")
    return b"".join(
        json.dumps(
            {
                "instruction": f"Question {index}",
                "context": "",
                "response": f"Answer {index}",
                "category": categories[index % 3],
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for index in range(500)
    )


def write_inputs(tmp_path: Path):
    payload = snapshot()
    source_path = tmp_path / "source.jsonl"
    source_path.write_bytes(payload)
    revision = "b" * 40
    source = SFTSourceSpec(
        "databricks/databricks-dolly-15k",
        revision,
        "databricks-dolly-15k.jsonl",
        sha256(payload).hexdigest(),
        len(payload),
        "CC-BY-SA-3.0",
        "https://creativecommons.org/licenses/by-sa/3.0/",
        "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
        f"resolve/{revision}/databricks-dolly-15k.jsonl",
    )
    policy = SFTDatasetPolicy(example_count=30, validation_count=6)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"source": asdict(source), "policy": policy.as_mapping()})
    )
    return source_path, config, source, policy


def test_build_command_publishes_verified_dataset_and_summary(tmp_path: Path, capsys):
    source_path, config, _, _ = write_inputs(tmp_path)
    output = tmp_path / "evidence"
    assert (
        main(
            [
                "build",
                "--source",
                str(source_path),
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["examples"] == 30
    assert summary["train_examples"] == 24
    assert summary["validation_examples"] == 6
    assert verify_sft_dataset(output)["manifest_sha256"] == summary["manifest_sha256"]


def test_config_loader_round_trips_strict_source_and_policy(tmp_path: Path):
    _, config, source, policy = write_inputs(tmp_path)
    assert load_build_config(config) == (source, policy)


def test_build_command_rejects_alias_and_existing_output(tmp_path: Path):
    source_path, config, _, _ = write_inputs(tmp_path)
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "build",
                "--source",
                str(source_path),
                "--config",
                str(config),
                "--output",
                str(source_path),
            ]
        )
    output = tmp_path / "evidence"
    output.mkdir()
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "build",
                "--source",
                str(source_path),
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )


def test_config_loader_rejects_duplicate_or_missing_fields(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text('{"source":{},"source":{},"policy":{}}')
    with pytest.raises(ValueError, match="duplicate"):
        load_build_config(path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="source and policy"):
        load_build_config(path)
