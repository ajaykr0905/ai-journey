from copy import deepcopy
from hashlib import sha256

from ai_journey.sft_dataset_build import SFTDatasetPolicy, build_sft_dataset
from ai_journey.sft_dataset_identity import canonical_json_bytes
from ai_journey.sft_dataset_record import DollyRecord
from ai_journey.sft_dataset_report import build_sft_manifest, manifest_sha256
from ai_journey.sft_dataset_source import SFTSourceSpec


def records():
    categories = ("open_qa", "classification", "summarization")
    return tuple(
        DollyRecord(f"Question {i}", "", f"Answer {i}", categories[i % 3])
        for i in range(40)
    )


def source():
    return SFTSourceSpec(
        dataset_id="databricks/databricks-dolly-15k",
        revision="b" * 40,
        filename="databricks-dolly-15k.jsonl",
        sha256="a" * 64,
        bytes=1_000,
        license="CC-BY-SA-3.0",
        license_url="https://creativecommons.org/licenses/by-sa/3.0/",
        source_url=(
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
            f"resolve/{'b' * 40}/databricks-dolly-15k.jsonl"
        ),
    )


def manifest():
    policy = SFTDatasetPolicy(example_count=30, validation_count=6)
    return build_sft_manifest(source(), policy, build_sft_dataset(records(), policy))


def test_manifest_binds_source_policy_pipeline_metrics_and_artifacts():
    value = manifest()
    assert value["schema_version"] == 1
    assert value["source"]["license"] == "CC-BY-SA-3.0"
    assert value["policy"]["example_count"] == 30
    assert value["pipeline"]["selection_candidate_records"] == 40
    assert value["metrics"]["examples"] == 30
    assert set(value["artifacts"]) == {"dataset.jsonl", "chatml.jsonl"}
    assert "Databricks" in value["attribution"]
    assert "no claim" in value["evidence_boundary"]


def test_manifest_seal_is_canonical_and_covers_every_other_field():
    value = manifest()
    unsealed = {key: item for key, item in value.items() if key != "manifest_sha256"}
    assert (
        value["manifest_sha256"] == sha256(canonical_json_bytes(unsealed)).hexdigest()
    )
    assert manifest_sha256(value) == value["manifest_sha256"]
    changed = deepcopy(value)
    changed["metrics"]["examples"] += 1
    assert manifest_sha256(changed) != value["manifest_sha256"]


def test_manifest_is_deterministic_for_the_same_build():
    assert manifest() == manifest()
