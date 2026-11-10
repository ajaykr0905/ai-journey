from hashlib import sha256
import json
from pathlib import Path

import pytest

from ai_journey.sft_dataset_build import SFTDatasetPolicy, build_sft_dataset
from ai_journey.sft_dataset_io import publish_sft_dataset
from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError
from ai_journey.sft_dataset_replay import replay_sft_dataset
from ai_journey.sft_dataset_source import SFTSourceSpec, load_dolly_snapshot


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


def source(payload: bytes) -> SFTSourceSpec:
    revision = "b" * 40
    return SFTSourceSpec(
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


def evidence(tmp_path: Path):
    payload = snapshot()
    source_path = tmp_path / "source.jsonl"
    source_path.write_bytes(payload)
    spec = source(payload)
    records = load_dolly_snapshot(source_path, spec)
    policy = SFTDatasetPolicy(example_count=30, validation_count=6)
    build = build_sft_dataset(records, policy)
    output = tmp_path / "evidence"
    publish_sft_dataset(output, spec, policy, build)
    return source_path, output


def test_replay_rebuilds_exact_manifest_and_artifact_bytes(tmp_path: Path):
    source_path, output = evidence(tmp_path)
    manifest = replay_sft_dataset(source_path, output)
    assert manifest["metrics"]["examples"] == 30


def test_replay_rejects_source_tampering_before_construction(tmp_path: Path):
    source_path, output = evidence(tmp_path)
    payload = source_path.read_bytes()
    source_path.write_bytes(payload.replace(b"Question 0", b"Question X", 1))
    with pytest.raises(SFTDatasetError, match="identity"):
        replay_sft_dataset(source_path, output)


def test_replay_requires_path_values(tmp_path: Path):
    source_path, output = evidence(tmp_path)
    with pytest.raises(TypeError, match="pathlib.Path"):
        replay_sft_dataset(str(source_path), output)


def test_replay_rejects_evidence_from_a_different_valid_source(tmp_path: Path):
    source_path, output = evidence(tmp_path)
    other = tuple(
        DollyRecord(f"Other {index}", "", f"Answer {index}", "open_qa")
        for index in range(500)
    )
    assert (
        other[0].instruction
        != load_dolly_snapshot(source_path, source(snapshot()))[0].instruction
    )
    with pytest.raises(SFTDatasetError):
        replay_sft_dataset(tmp_path / "missing.jsonl", output)
