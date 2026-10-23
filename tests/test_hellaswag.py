import copy
import json

import pytest
import torch

from ai_journey.hellaswag import HellaSwagError, HellaSwagExample, HellaSwagSource
from ai_journey.hellaswag import records_from_dataset_server
from ai_journey.hellaswag import records_digest, select_records
from ai_journey.hellaswag import encode_example
from ai_journey.hellaswag import score_record
from ai_journey.hellaswag import probabilities_from_scores
from ai_journey.hellaswag import evaluate_records
from ai_journey.hellaswag import split_type_metrics
from ai_journey.hellaswag import bootstrap_accuracy
from ai_journey.hellaswag import shard_records
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel


def example_payload(index=24, label="3"):
    return {
        "ind": index,
        "activity_label": "Roof shingle removal",
        "ctx_a": "A man is sitting on a roof.",
        "ctx_b": "he",
        "ctx": "A man is sitting on a roof. he",
        "endings": [
            "is using wrap to wrap a pair of skis.",
            "is ripping level tiles off.",
            "is holding a rubik's cube.",
            "starts pulling up roofing on a roof.",
        ],
        "source_id": "activitynet~public-example",
        "split": "val",
        "split_type": "indomain",
        "label": label,
    }


def test_example_schema_normalizes_label_and_has_stable_digest():
    first = HellaSwagExample.from_mapping(example_payload())
    second = HellaSwagExample.from_mapping(example_payload(label=3))
    assert first == second
    assert first.label == 3
    assert first.canonical_context == "A man is sitting on a roof. he"
    assert len(first.digest) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ind", True),
        ("activity_label", ""),
        ("endings", ["one", "two", "three"]),
        ("endings", ["same"] * 4),
        ("label", "4"),
        ("ctx", "A different context"),
        ("split", "train"),
        ("split_type", "unknown"),
    ],
)
def test_example_schema_rejects_malformed_records(field, value):
    payload = copy.deepcopy(example_payload())
    payload[field] = value
    with pytest.raises(HellaSwagError):
        HellaSwagExample.from_mapping(payload)


def test_example_schema_rejects_unknown_fields():
    payload = example_payload()
    payload["unexpected"] = "field"
    with pytest.raises(HellaSwagError, match="fields"):
        HellaSwagExample.from_mapping(payload)


def source_payload():
    return {
        "repository": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "validation_url": "https://huggingface.co/example.parquet",
        "validation_sha256": "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd",
        "validation_bytes": 6315951,
        "validation_rows": 10042,
        "license": "MIT",
    }


def test_source_contract_pins_public_validation_bytes():
    source = HellaSwagSource.from_mapping(source_payload())
    assert source.validation_rows == 10042
    assert source.license == "MIT"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "main"),
        ("validation_sha256", "not-a-digest"),
        ("validation_bytes", True),
        ("validation_rows", 0),
        ("license", "unknown"),
    ],
)
def test_source_contract_rejects_unpinned_or_invalid_metadata(field, value):
    payload = source_payload()
    payload[field] = value
    with pytest.raises(HellaSwagError):
        HellaSwagSource.from_mapping(payload)


def server_payload(*rows):
    return {
        "dataset": "Rowan/hellaswag",
        "config": "default",
        "split": "validation",
        "features": [],
        "rows": [
            {"row_idx": offset, "row": row, "truncated_cells": []}
            for offset, row in enumerate(rows)
        ],
        "truncated": True,
    }


def test_dataset_server_loader_preserves_validated_row_identities():
    payload = server_payload(example_payload())
    records = records_from_dataset_server(json.dumps(payload).encode())
    assert [record.row_index for record in records] == [0]
    assert [record.example.index for record in records] == [24]
    assert records[0].example.label == 3


@pytest.mark.parametrize("mutation", ["dataset", "row-index", "truncated-cell"])
def test_dataset_server_loader_rejects_mismatched_envelopes(mutation):
    payload = server_payload(example_payload())
    if mutation == "dataset":
        payload["dataset"] = "other/dataset"
    elif mutation == "row-index":
        payload["rows"][0]["row_idx"] = -1
    else:
        payload["rows"][0]["truncated_cells"] = ["ctx"]
    with pytest.raises(HellaSwagError):
        records_from_dataset_server(json.dumps(payload).encode())


def test_dataset_server_loader_rejects_duplicate_rows():
    row = example_payload()
    with pytest.raises(HellaSwagError, match="duplicate"):
        records_from_dataset_server(json.dumps(server_payload(row, row)).encode())


def test_record_selection_requires_complete_contiguous_coverage():
    rows = [example_payload(index=100 + offset) for offset in range(4)]
    records = records_from_dataset_server(json.dumps(server_payload(*rows)).encode())
    selected = select_records(reversed(records), start=1, count=2)
    assert [record.row_index for record in selected] == [1, 2]
    assert records_digest(selected) == records_digest(tuple(selected))
    assert len(records_digest(selected)) == 64
    with pytest.raises(HellaSwagError, match="cover"):
        select_records(records, start=2, count=3)


@pytest.mark.parametrize(("start", "count"), [(True, 1), (-1, 1), (0, 0), (0, 1001)])
def test_record_selection_enforces_evaluation_budgets(start, count):
    records = records_from_dataset_server(
        json.dumps(server_payload(example_payload())).encode()
    )
    with pytest.raises(HellaSwagError, match="budget"):
        select_records(records, start=start, count=count)


def test_example_encoding_uses_utf8_and_one_continuation_boundary_space():
    payload = example_payload()
    payload["endings"][3] = "  starts removing shingles."
    example = HellaSwagExample.from_mapping(payload)
    prompt, endings = encode_example(example)
    assert prompt.decode() == example.context
    assert endings[3] == b" starts removing shingles."
    assert all(
        ending.startswith(b" ") and not ending.startswith(b"  ") for ending in endings
    )


@pytest.mark.parametrize("limits", [{"max_context_bytes": 1}, {"max_ending_bytes": 1}])
def test_example_encoding_enforces_byte_budgets(limits):
    with pytest.raises(HellaSwagError, match="budget"):
        encode_example(HellaSwagExample.from_mapping(example_payload()), **limits)


def test_record_scoring_excludes_prompt_and_reports_stable_tie_break():
    record = records_from_dataset_server(
        json.dumps(server_payload(example_payload())).encode()
    )[0]
    model = GPT2LanguageModel(GPT2Config(256, 8, 8, 2, 1)).double().train()
    model.ln_f.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result = score_record(model, record)
    assert result["prediction"] == 0
    assert not result["correct"]
    assert result["label"] == 3
    assert result["probabilities"] == pytest.approx([0.25] * 4)
    assert result["confidence"] == pytest.approx(0.25)
    assert [score["tokens"] for score in result["scores"]] == [
        len(ending) for ending in encode_example(record.example)[1]
    ]
    assert model.training and not model.ln_f.training


def test_score_probabilities_are_stable_for_large_candidate_losses():
    probabilities = probabilities_from_scores(
        [{"mean_nll": value} for value in (10000.0, 10001.0, 10002.0, 10003.0)]
    )
    assert sum(probabilities) == pytest.approx(1)
    assert probabilities[0] > probabilities[1] > probabilities[2] > probabilities[3]
    with pytest.raises(HellaSwagError):
        probabilities_from_scores([{"mean_nll": float("nan")}] * 4)


def test_evaluation_aggregates_accuracy_nll_and_calibration_once_per_row():
    payloads = [
        example_payload(index=24, label="0"),
        example_payload(index=25, label="3"),
    ]
    payloads[1]["source_id"] = "activitynet~second"
    records = records_from_dataset_server(
        json.dumps(server_payload(*payloads)).encode()
    )
    model = GPT2LanguageModel(GPT2Config(256, 8, 8, 2, 1)).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result = evaluate_records(model, records)
    assert result["examples"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == 0.5
    assert result["accuracy_interval"]["lower"] == 0
    assert result["accuracy_interval"]["upper"] == 1
    assert result["mean_correct_nll"] == pytest.approx(torch.log(torch.tensor(256.0)))
    assert result["calibration"]["brier_score"] == pytest.approx(0.75)
    assert result["split_types"]["indomain"]["examples"] == 2
    assert result["split_types"]["zeroshot"] is None
    assert [row["row_index"] for row in result["records"]] == [0, 1]


def test_evaluation_rejects_empty_or_duplicate_rows():
    with pytest.raises(HellaSwagError):
        evaluate_records(object(), [])
    record = records_from_dataset_server(
        json.dumps(server_payload(example_payload())).encode()
    )[0]
    with pytest.raises(HellaSwagError, match="unique"):
        evaluate_records(object(), [record, record])


def test_split_metrics_keep_indomain_and_zeroshot_results_separate():
    results = [
        {
            "split_type": split_type,
            "correct": correct,
            "label": 0,
            "scores": [{"mean_nll": loss}] * 4,
        }
        for split_type, correct, loss in (
            ("indomain", True, 1.0),
            ("zeroshot", False, 3.0),
        )
    ]
    metrics = split_type_metrics(results)
    assert metrics["indomain"] == {
        "examples": 1,
        "correct": 1,
        "accuracy": 1.0,
        "mean_correct_nll": 1.0,
    }
    assert metrics["zeroshot"]["accuracy"] == 0


def test_accuracy_bootstrap_is_deterministic_and_bounded():
    mixed = [{"correct": value} for value in (True, False, True, False)]
    first = bootstrap_accuracy(mixed, samples=200, seed=7)
    assert first == bootstrap_accuracy(mixed, samples=200, seed=7)
    assert 0 <= first["lower"] <= first["upper"] <= 1
    perfect = bootstrap_accuracy([{"correct": True}] * 3, samples=100)
    assert perfect["lower"] == perfect["upper"] == 1
    with pytest.raises(HellaSwagError, match="budget"):
        bootstrap_accuracy(mixed, samples=99)


def test_stable_shards_cover_each_validation_row_exactly_once():
    rows = []
    for offset in range(7):
        row = example_payload(index=100 + offset, label=str(offset % 4))
        row["source_id"] = f"activitynet~{offset}"
        rows.append(row)
    records = records_from_dataset_server(json.dumps(server_payload(*rows)).encode())
    shards = [
        shard_records(reversed(records), rank=rank, world_size=3) for rank in range(3)
    ]
    assert [[record.row_index for record in shard] for shard in shards] == [
        [0, 3, 6],
        [1, 4],
        [2, 5],
    ]
    assert sorted(record.row_index for shard in shards for record in shard) == list(
        range(7)
    )
    with pytest.raises(HellaSwagError, match="topology"):
        shard_records(records, rank=3, world_size=3)
