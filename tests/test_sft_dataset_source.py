from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import pytest

from ai_journey.sft_dataset_record import SFTDatasetError
from ai_journey.sft_dataset_source import (
    SFTSourceSpec,
    load_dolly_snapshot,
    load_source_spec,
)


REVISION = "b" * 40


def source_spec(payload: bytes) -> SFTSourceSpec:
    return SFTSourceSpec(
        dataset_id="databricks/databricks-dolly-15k",
        revision=REVISION,
        filename="databricks-dolly-15k.jsonl",
        sha256=sha256(payload).hexdigest(),
        bytes=len(payload),
        license="CC-BY-SA-3.0",
        license_url="https://creativecommons.org/licenses/by-sa/3.0/",
        source_url=(
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
            f"resolve/{REVISION}/databricks-dolly-15k.jsonl"
        ),
    )


def snapshot(count=500) -> bytes:
    return b"".join(
        json.dumps(
            {
                "instruction": f"Instruction {index}",
                "context": "",
                "response": f"Response {index}",
                "category": "open_qa",
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for index in range(count)
    )


def test_policy_round_trips_from_strict_json(tmp_path: Path):
    payload = snapshot()
    spec = source_spec(payload)
    path = tmp_path / "source.json"
    path.write_text(json.dumps(asdict(spec)))
    assert load_source_spec(path) == spec


def test_snapshot_verifies_bytes_and_loads_normalized_records(tmp_path: Path):
    payload = snapshot()
    path = tmp_path / "dolly.jsonl"
    path.write_bytes(payload)
    records = load_dolly_snapshot(path, source_spec(payload))
    assert len(records) == 500
    assert records[0].instruction == "Instruction 0"
    assert records[-1].response == "Response 499"


def test_snapshot_accepts_a_valid_final_jsonl_record_without_newline(tmp_path: Path):
    payload = snapshot()[:-1]
    path = tmp_path / "dolly.jsonl"
    path.write_bytes(payload)
    records = load_dolly_snapshot(path, source_spec(payload))
    assert len(records) == 500
    assert records[-1].instruction == "Instruction 499"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload: payload[:-1], "identity"),
        (
            lambda payload: payload.replace(b"Instruction 0", b"Instruction X", 1),
            "identity",
        ),
    ],
)
def test_snapshot_rejects_truncation_and_tampering(tmp_path: Path, mutation, match):
    payload = snapshot()
    path = tmp_path / "dolly.jsonl"
    path.write_bytes(mutation(payload))
    with pytest.raises(SFTDatasetError, match=match):
        load_dolly_snapshot(path, source_spec(payload))


def test_snapshot_rejects_duplicate_fields_and_too_few_records(tmp_path: Path):
    duplicate = (
        b'{"instruction":"one","instruction":"two","context":"",'
        b'"response":"answer","category":"open_qa"}\n'
    )
    payload = duplicate * 500
    path = tmp_path / "dolly.jsonl"
    path.write_bytes(payload)
    with pytest.raises(SFTDatasetError, match="duplicate JSON field"):
        load_dolly_snapshot(path, source_spec(payload))
    small = snapshot(499)
    path.write_bytes(small)
    with pytest.raises(SFTDatasetError, match="record count"):
        load_dolly_snapshot(path, source_spec(small))


def test_source_policy_rejects_unpinned_or_misattributed_inputs():
    payload = snapshot()
    values = asdict(source_spec(payload))
    for field, value in (
        ("revision", "main"),
        ("license", "unknown"),
        ("source_url", "https://example.com/data.jsonl"),
        ("bytes", True),
    ):
        changed = {**values, field: value}
        with pytest.raises(SFTDatasetError):
            SFTSourceSpec.from_mapping(changed)
