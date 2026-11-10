from copy import deepcopy

import pytest

from ai_journey.sft_dataset_identity import record_id
from ai_journey.sft_dataset_metrics import dataset_metrics, validate_dataset_metrics
from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError
from ai_journey.sft_dataset_split import SFTDatasetSplit


def record(category, index, context=""):
    return DollyRecord(f"Question {index}", context, f"Answer {index}", category)


def split():
    train = tuple(
        sorted((record("open_qa", 1), record("open_qa", 2, "ctx")), key=record_id)
    )
    validation = (record("classification", 3),)
    return SFTDatasetSplit(train, validation)


def test_dataset_metrics_reconcile_partitions_categories_context_and_bytes():
    metrics = dataset_metrics(split())
    assert validate_dataset_metrics(metrics) is metrics
    assert metrics["examples"] == 3
    assert metrics["train_examples"] == 2
    assert metrics["validation_examples"] == 1
    assert metrics["context_examples"] == 1
    assert metrics["category_examples"] == {"classification": 1, "open_qa": 2}
    assert metrics["assistant_utf8_bytes"]["total"] == 24


def test_prompt_metrics_include_the_explicit_context_delimiter():
    metrics = dataset_metrics(split())
    expected = len("Question 2\n\nContext:\nctx".encode())
    assert metrics["prompt_utf8_bytes"]["max"] == expected


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(examples=4),
        lambda value: value["category_examples"].update(open_qa=3),
        lambda value: value["assistant_utf8_bytes"].update(mean=float("nan")),
        lambda value: value["prompt_utf8_bytes"].update(total=0),
    ],
)
def test_metric_validator_rejects_tampered_evidence(mutation):
    value = deepcopy(dataset_metrics(split()))
    mutation(value)
    with pytest.raises(SFTDatasetError):
        validate_dataset_metrics(value)


def test_metrics_require_a_validated_split():
    with pytest.raises(TypeError, match="SFTDatasetSplit"):
        dataset_metrics({})
