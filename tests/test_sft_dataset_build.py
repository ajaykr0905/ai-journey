from dataclasses import replace

import pytest

from ai_journey.sft_dataset_build import (
    SFTDatasetPolicy,
    build_sft_dataset,
)
from ai_journey.sft_dataset_quality import SFTQualityPolicy
from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError


CATEGORIES = ("open_qa", "classification", "summarization")


def records(count=60):
    return tuple(
        DollyRecord(
            f"Question {index}",
            "context" if index % 2 else "",
            f"Answer {index}",
            CATEGORIES[index % len(CATEGORIES)],
        )
        for index in range(count)
    )


def policy():
    return SFTDatasetPolicy(example_count=30, validation_count=6)


def test_build_composes_an_exact_reproducible_dataset():
    result = build_sft_dataset(records(), policy())
    assert result.metrics["examples"] == 30
    assert result.metrics["train_examples"] == 24
    assert result.metrics["validation_examples"] == 6
    assert len(result.artifacts.dataset_jsonl.splitlines()) == 30
    assert len(result.artifacts.chatml_jsonl.splitlines()) == 30
    assert result == build_sft_dataset(records(), policy())


def test_build_reports_quality_and_duplicate_removals_before_selection():
    rows = records()
    duplicate = rows[0]
    identity = DollyRecord("repeat", "", "REPEAT", "open_qa")
    result = build_sft_dataset(rows + (duplicate, identity), policy())
    assert result.quality.rejection_counts == {"identity_answer": 1}
    assert result.dedup.duplicate_records == 1
    assert result.selection.candidate_records == len(rows)


def test_build_rejects_stages_whose_record_counts_disagree():
    result = build_sft_dataset(records(), policy())
    other = build_sft_dataset(records(61), policy())
    with pytest.raises(SFTDatasetError, match="selection counts do not reconcile"):
        replace(result, selection=other.selection)


def test_policy_round_trips_all_explicit_controls():
    value = SFTDatasetPolicy(
        example_count=30,
        validation_count=6,
        selection_seed=11,
        split_seed=12,
        quality=SFTQualityPolicy(min_response_bytes=2),
    )
    assert SFTDatasetPolicy.from_mapping(value.as_mapping()) == value


@pytest.mark.parametrize(
    "changes",
    [
        {"example_count": 7},
        {"validation_count": 30},
        {"selection_seed": True},
        {"split_seed": 2**32},
        {"quality": {}},
    ],
)
def test_policy_rejects_invalid_controls(changes):
    with pytest.raises(SFTDatasetError):
        replace(policy(), **changes)


def test_build_rejects_unvalidated_policy():
    with pytest.raises(TypeError, match="SFTDatasetPolicy"):
        build_sft_dataset(records(), {})
