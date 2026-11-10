from copy import deepcopy
import json
from pathlib import Path

import pytest

from ai_journey.sft_dataset_build import SFTDatasetPolicy, build_sft_dataset
from ai_journey.sft_dataset_identity import canonical_json_bytes
from ai_journey.sft_dataset_io import publish_sft_dataset
from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError
from ai_journey.sft_dataset_report import manifest_sha256
from ai_journey.sft_dataset_source import SFTSourceSpec
from ai_journey.sft_dataset_verify import verify_sft_dataset


def inputs():
    categories = ("open_qa", "classification", "summarization")
    records = tuple(
        DollyRecord(f"Question {i}", "", f"Answer {i}", categories[i % 3])
        for i in range(40)
    )
    policy = SFTDatasetPolicy(example_count=30, validation_count=6)
    revision = "b" * 40
    source = SFTSourceSpec(
        "databricks/databricks-dolly-15k",
        revision,
        "databricks-dolly-15k.jsonl",
        "a" * 64,
        1_000,
        "CC-BY-SA-3.0",
        "https://creativecommons.org/licenses/by-sa/3.0/",
        "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
        f"resolve/{revision}/databricks-dolly-15k.jsonl",
    )
    return source, policy, build_sft_dataset(records, policy)


def evidence(tmp_path: Path):
    source, policy, build = inputs()
    output = tmp_path / "evidence"
    publish_sft_dataset(output, source, policy, build)
    return output


def test_verifier_accepts_complete_published_evidence(tmp_path: Path):
    output = evidence(tmp_path)
    manifest = verify_sft_dataset(output)
    assert manifest["metrics"]["examples"] == 30
    assert manifest_sha256(manifest) == manifest["manifest_sha256"]


def test_verifier_rejects_manifest_and_artifact_byte_tampering(tmp_path: Path):
    output = evidence(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["metrics"]["examples"] += 1
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(SFTDatasetError, match="manifest"):
        verify_sft_dataset(output)

    output = evidence(tmp_path / "second")
    (output / "dataset.jsonl").write_bytes(
        (output / "dataset.jsonl").read_bytes() + b"{}\n"
    )
    with pytest.raises(SFTDatasetError, match="artifact"):
        verify_sft_dataset(output)


def test_verifier_rejects_semantic_chatml_tampering_even_when_resealed(tmp_path: Path):
    output = evidence(tmp_path)
    rows = [
        json.loads(line) for line in (output / "chatml.jsonl").read_bytes().splitlines()
    ]
    rows[0]["text"] = rows[0]["text"].replace("Answer", "Altered", 1)
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    (output / "chatml.jsonl").write_bytes(payload)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["artifacts"]["chatml.jsonl"] = {
        "bytes": len(payload),
        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(SFTDatasetError, match="does not match"):
        verify_sft_dataset(output)


def test_verifier_rejects_extra_inventory_and_non_directory_inputs(tmp_path: Path):
    output = evidence(tmp_path)
    (output / "extra.txt").write_text("unexpected")
    with pytest.raises(SFTDatasetError, match="inventory"):
        verify_sft_dataset(output)
    with pytest.raises(TypeError, match="pathlib.Path"):
        verify_sft_dataset("evidence")


def test_verifier_rejects_resealed_pipeline_count_tampering(tmp_path: Path):
    output = evidence(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    changed = deepcopy(manifest)
    changed["pipeline"]["input_records"] += 1
    changed["manifest_sha256"] = manifest_sha256(changed)
    (output / "manifest.json").write_bytes(canonical_json_bytes(changed))
    with pytest.raises(SFTDatasetError, match="pipeline"):
        verify_sft_dataset(output)
