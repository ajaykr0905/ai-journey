import json

import pytest

from ai_journey.sft_dataset_export import SFTDatasetArtifacts, export_split
from ai_journey.sft_dataset_identity import record_id
from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError
from ai_journey.sft_dataset_split import SFTDatasetSplit


def record(index):
    return DollyRecord(f"Question {index}", "", f"Answer {index}", "open_qa")


def split():
    train = tuple(sorted((record(1), record(2)), key=record_id))
    return SFTDatasetSplit(train, (record(3),))


def rows(payload):
    return [json.loads(line) for line in payload.splitlines()]


def test_export_writes_standard_messages_and_explicit_splits():
    artifacts = export_split(split())
    dataset = rows(artifacts.dataset_jsonl)
    assert len(dataset) == 3
    assert [row["split"] for row in dataset] == ["train", "train", "validation"]
    assert [message["role"] for message in dataset[0]["messages"]] == [
        "user",
        "assistant",
    ]


def test_chatml_export_binds_identity_text_and_assistant_spans():
    artifacts = export_split(split())
    rendered = rows(artifacts.chatml_jsonl)
    for row in rendered:
        start, end = row["assistant_spans"][0]
        assert row["text"][start:end].startswith('"Answer ')
        assert row["text"].endswith("<|end|>\n")
    assert {row["example_id"] for row in rendered} == {
        row["example_id"] for row in rows(artifacts.dataset_jsonl)
    }


def test_export_is_deterministic_and_describes_exact_bytes():
    first = export_split(split())
    assert first == export_split(split())
    descriptors = first.descriptors()
    assert descriptors["dataset.jsonl"]["bytes"] == len(first.dataset_jsonl)
    assert len(descriptors["chatml.jsonl"]["sha256"]) == 64


def test_artifact_contract_rejects_empty_unterminated_or_unbounded_payloads():
    for payload in (b"", b"{}", b"\x00\n", b"x" * (64 * 1024 * 1024 + 1) + b"\n"):
        with pytest.raises(SFTDatasetError):
            SFTDatasetArtifacts(payload, b"{}\n")


def test_export_requires_a_validated_split():
    with pytest.raises(TypeError, match="SFTDatasetSplit"):
        export_split({})
