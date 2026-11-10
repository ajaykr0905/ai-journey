from pathlib import Path
from unittest.mock import patch

import pytest

from ai_journey.sft_dataset_build import SFTDatasetPolicy, build_sft_dataset
from ai_journey.sft_dataset_io import publish_sft_dataset
from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError
from ai_journey.sft_dataset_report import manifest_sha256
from ai_journey.sft_dataset_source import SFTSourceSpec


def records():
    categories = ("open_qa", "classification", "summarization")
    return tuple(
        DollyRecord(f"Question {i}", "", f"Answer {i}", categories[i % 3])
        for i in range(40)
    )


def source():
    revision = "b" * 40
    return SFTSourceSpec(
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


def inputs():
    policy = SFTDatasetPolicy(example_count=30, validation_count=6)
    return policy, build_sft_dataset(records(), policy)


def test_publication_atomically_writes_exact_inventory(tmp_path: Path):
    policy, build = inputs()
    output = tmp_path / "evidence"
    manifest = publish_sft_dataset(output, source(), policy, build)
    assert {path.name for path in output.iterdir()} == {
        "dataset.jsonl",
        "chatml.jsonl",
        "manifest.json",
    }
    assert (
        output.joinpath("dataset.jsonl").read_bytes() == build.artifacts.dataset_jsonl
    )
    assert manifest_sha256(manifest) == manifest["manifest_sha256"]


def test_publication_refuses_to_replace_existing_output(tmp_path: Path):
    policy, build = inputs()
    output = tmp_path / "evidence"
    output.mkdir()
    marker = output / "owned.txt"
    marker.write_text("preserve")
    with pytest.raises(SFTDatasetError, match="already exists"):
        publish_sft_dataset(output, source(), policy, build)
    assert marker.read_text() == "preserve"


def test_failed_publication_leaves_no_partial_destination(tmp_path: Path):
    policy, build = inputs()
    output = tmp_path / "evidence"
    with (
        patch("ai_journey.sft_dataset_io.os.replace", side_effect=OSError("stop")),
        pytest.raises(OSError, match="stop"),
    ):
        publish_sft_dataset(output, source(), policy, build)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".evidence.*"))


def test_publication_requires_a_path_destination():
    policy, build = inputs()
    with pytest.raises(TypeError, match="pathlib.Path"):
        publish_sft_dataset("output", source(), policy, build)
