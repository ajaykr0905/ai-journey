import copy
from hashlib import sha256
import json

import pytest

from ai_journey.lambada import LambadaError, LambadaExample, LambadaSource
from ai_journey.lambada import examples_from_dataset_server


def source_payload(sample=b"bounded public response"):
    return {
        "repository": "EleutherAI/lambada_openai",
        "revision": "900124bf3b8235c6daf21033af9948b3f07346c4",
        "dataset_server_url": "https://datasets-server.huggingface.co/first-rows",
        "sample_sha256": sha256(sample).hexdigest(),
        "sample_bytes": len(sample),
        "test_rows": 5153,
        "source_file_sha256": (
            "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226"
        ),
        "license": "MIT",
    }


def test_source_contract_pins_public_snapshot():
    sample = b"bounded public response"
    source = LambadaSource.from_mapping(source_payload(sample))
    source.verify_sample(sample)
    assert source.test_rows == 5153


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "other/dataset"),
        ("revision", "not-a-digest"),
        ("sample_bytes", True),
        ("test_rows", 0),
        ("license", "unknown"),
    ],
)
def test_source_contract_rejects_unpinned_metadata(field, value):
    payload = source_payload()
    payload[field] = value
    with pytest.raises(LambadaError):
        LambadaSource.from_mapping(payload)


def test_source_contract_rejects_changed_sample_bytes():
    source = LambadaSource.from_mapping(source_payload())
    with pytest.raises(LambadaError, match="snapshot"):
        source.verify_sample(b"changed")


def test_example_has_stable_content_identity():
    first = LambadaExample(7, "A short public-safe fixture ends here")
    second = LambadaExample(7, "A short public-safe fixture ends here")
    assert first.digest == second.digest
    assert len(first.digest) == 64


@pytest.mark.parametrize(
    ("row_index", "text"),
    [(-1, "valid text"), (True, "valid text"), (0, ""), (0, "contains\x00null")],
)
def test_example_rejects_invalid_or_unsafe_values(row_index, text):
    with pytest.raises(LambadaError):
        LambadaExample(row_index, text)


def server_payload():
    return {
        "dataset": "EleutherAI/lambada_openai",
        "config": "default",
        "split": "test",
        "features": [{"feature_idx": 0, "name": "text", "type": {"dtype": "string"}}],
        "rows": [
            {
                "row_idx": 7,
                "row": {"text": "A bounded fixture predicts the final word"},
                "truncated_cells": [],
            },
            {
                "row_idx": 8,
                "row": {"text": "Another original fixture ends safely"},
                "truncated_cells": [],
            },
        ],
        "truncated": True,
    }


def test_dataset_response_parses_bounded_test_rows():
    examples = examples_from_dataset_server(json.dumps(server_payload()).encode())
    assert [example.row_index for example in examples] == [7, 8]
    assert examples[0].text.endswith("final word")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", "other/dataset"),
        ("config", "en"),
        ("split", "train"),
        ("rows", []),
        ("truncated", "yes"),
    ],
)
def test_dataset_response_rejects_wrong_metadata(field, value):
    payload = server_payload()
    payload[field] = value
    with pytest.raises(LambadaError, match="metadata"):
        examples_from_dataset_server(json.dumps(payload).encode())


def test_dataset_response_rejects_truncated_or_duplicate_rows():
    payload = server_payload()
    payload["rows"][0]["truncated_cells"] = ["text"]
    with pytest.raises(LambadaError, match="truncated"):
        examples_from_dataset_server(json.dumps(payload).encode())
    duplicate = copy.deepcopy(server_payload())
    duplicate["rows"].append(copy.deepcopy(duplicate["rows"][0]))
    with pytest.raises(LambadaError, match="duplicate"):
        examples_from_dataset_server(json.dumps(duplicate).encode())


def test_dataset_response_rejects_invalid_json_and_oversized_inputs():
    with pytest.raises(LambadaError, match="JSON"):
        examples_from_dataset_server(b"not-json")
    with pytest.raises(LambadaError, match="budget"):
        examples_from_dataset_server(b"x" * (2 * 1024 * 1024 + 1))
