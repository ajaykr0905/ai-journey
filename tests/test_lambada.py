import copy
from hashlib import sha256
import json

import pytest
import torch

from ai_journey.lambada import LambadaError, LambadaExample, LambadaProtocol
from ai_journey.lambada import LambadaSource, examples_from_dataset_server
from ai_journey.lambada import encode_example, protocol_mapping, score_target
from ai_journey.lambada import select_examples
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel


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


def protocol_payload():
    return {
        "schema_version": 1,
        "source": source_payload(),
        "start": 7,
        "count": 2,
        "bootstrap_samples": 2000,
        "bootstrap_seed": 63,
        "max_context_bytes": 4096,
        "max_target_bytes": 64,
    }


def test_protocol_round_trips_and_has_stable_fingerprint():
    protocol = LambadaProtocol.from_mapping(protocol_payload())
    assert protocol_mapping(protocol) == protocol_payload()
    assert (
        protocol.fingerprint
        == LambadaProtocol.from_mapping(protocol_payload()).fingerprint
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start", -1),
        ("count", 1001),
        ("bootstrap_samples", 99),
        ("bootstrap_seed", True),
        ("max_context_bytes", 0),
        ("max_target_bytes", 257),
    ],
)
def test_protocol_rejects_controls_outside_budgets(field, value):
    payload = protocol_payload()
    payload[field] = value
    with pytest.raises(LambadaError, match="budget"):
        LambadaProtocol.from_mapping(payload)


def test_protocol_rejects_slice_past_test_rows():
    payload = protocol_payload()
    payload.update(start=5152, count=2)
    with pytest.raises(LambadaError, match="exceeds"):
        LambadaProtocol.from_mapping(payload)


def test_selection_returns_exact_ordered_slice():
    examples = examples_from_dataset_server(json.dumps(server_payload()).encode())
    selected = select_examples(reversed(examples), start=7, count=2)
    assert [item.row_index for item in selected] == [7, 8]


def test_selection_rejects_missing_or_invalid_rows():
    examples = examples_from_dataset_server(json.dumps(server_payload()).encode())
    with pytest.raises(LambadaError, match="cover"):
        select_examples(examples, start=6, count=2)
    with pytest.raises(LambadaError, match="validated"):
        select_examples([object()], start=7, count=1)


def test_example_encoding_preserves_context_separator_and_utf8_target():
    example = LambadaExample(4, "A café named its newest drink brûlée")
    item = encode_example(example)
    assert item.prompt == "A café named its newest drink ".encode()
    assert item.target == "brûlée".encode()
    assert item.example_sha256 == example.digest


@pytest.mark.parametrize(
    "text", ["singleword", "trailing whitespace ", "line without target\n"]
)
def test_example_encoding_rejects_ambiguous_last_word(text):
    with pytest.raises(LambadaError, match="final word"):
        encode_example(LambadaExample(0, text))


def test_example_encoding_enforces_context_and_target_budgets():
    example = LambadaExample(0, "context finalword")
    with pytest.raises(LambadaError, match="budget"):
        encode_example(example, max_context_bytes=2)
    with pytest.raises(LambadaError, match="budget"):
        encode_example(example, max_target_bytes=4)
    with pytest.raises(LambadaError, match="budget"):
        encode_example(example, max_target_bytes=True)


def test_target_scoring_is_finite_and_preserves_model_and_rng_state():
    model = GPT2LanguageModel(GPT2Config(256, 8, 8, 2, 1)).double().train()
    item = encode_example(LambadaExample(0, "public-safe context target"))
    torch.manual_seed(123)
    state = torch.get_rng_state().clone()
    result = score_target(model, item)
    assert result["target_bytes"] == len(b"target")
    assert result["loss_sum"] > 0
    assert result["bits_per_byte"] > 0
    assert model.training is True
    assert torch.equal(torch.get_rng_state(), state)


def test_target_scoring_rejects_non_items():
    model = GPT2LanguageModel(GPT2Config(256, 8, 8, 2, 1)).double()
    with pytest.raises(TypeError, match="LambadaItem"):
        score_target(model, object())
