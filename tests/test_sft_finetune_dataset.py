import json
from hashlib import sha256

import pytest

from ai_journey.sft_finetune_contract import SFTFineTuneError
from ai_journey.sft_finetune_dataset import load_chatml_dataset


def row(example_id, split, text="abc"):
    return {
        "assistant_spans": [[1, 3]],
        "example_id": example_id,
        "split": split,
        "text": text,
    }


def write_rows(path, rows):
    payload = b"".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for value in rows
    )
    path.write_bytes(payload)
    return payload


def test_load_chatml_dataset_preserves_order_partitions_and_digest(tmp_path):
    path = tmp_path / "chatml.jsonl"
    payload = write_rows(path, [row("train-1", "train"), row("valid-1", "validation")])
    dataset = load_chatml_dataset(path)
    assert [value.example_id for value in dataset.train] == ["train-1"]
    assert [value.example_id for value in dataset.validation] == ["valid-1"]
    assert dataset.sha256 == sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "rows",
    [
        [row("same", "train"), row("same", "validation")],
        [row("one", "train")],
        [row("one", "train", "a"), row("two", "validation")],
    ],
)
def test_load_chatml_dataset_rejects_invalid_partition_contracts(tmp_path, rows):
    path = tmp_path / "chatml.jsonl"
    write_rows(path, rows)
    with pytest.raises(SFTFineTuneError):
        load_chatml_dataset(path)


def test_load_chatml_dataset_rejects_symlinks_and_trailing_partial_rows(tmp_path):
    target = tmp_path / "target.jsonl"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(SFTFineTuneError):
        load_chatml_dataset(link)
    with pytest.raises(SFTFineTuneError):
        load_chatml_dataset(target)
