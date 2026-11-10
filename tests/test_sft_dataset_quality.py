from dataclasses import replace

import pytest

from ai_journey.sft_dataset_quality import (
    SFTQualityPolicy,
    filter_quality,
    quality_rejection,
)
from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError


def record(instruction="Question", context="", response="Answer"):
    return DollyRecord(instruction, context, response, "open_qa")


def test_quality_policy_accepts_bounded_nontrivial_examples():
    policy = SFTQualityPolicy()
    assert quality_rejection(record(), policy) is None
    result = filter_quality((record(), record(instruction="Second")), policy)
    assert len(result.accepted) == 2
    assert result.rejected == ()
    assert result.rejection_counts == {}


def test_quality_policy_explains_each_rejection_without_mutating_records():
    policy = SFTQualityPolicy(
        min_response_bytes=5,
        max_prompt_bytes=128,
        max_response_bytes=128,
    )
    rows = (
        record(instruction="same", response="SAME"),
        record(response="tiny"[:-1]),
        record(instruction="p" * 129),
        record(response="r" * 129),
        record(instruction="kept", response="valid"),
    )
    result = filter_quality(rows, policy)
    assert result.accepted == (rows[-1],)
    assert result.rejection_counts == {
        "identity_answer": 1,
        "prompt_too_large": 1,
        "response_too_large": 1,
        "response_too_small": 1,
    }


def test_context_prevents_false_identity_answer_rejection():
    value = record(instruction="repeat", context="requested context", response="repeat")
    assert quality_rejection(value, SFTQualityPolicy()) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"min_response_bytes": True},
        {"max_prompt_bytes": 127},
        {"max_response_bytes": 64 * 1024 + 1},
        {"reject_identity_answers": 1},
    ],
)
def test_quality_policy_rejects_invalid_controls(changes):
    with pytest.raises(SFTDatasetError):
        replace(SFTQualityPolicy(), **changes)


def test_quality_filter_rejects_invalid_or_empty_inputs():
    with pytest.raises(SFTDatasetError, match="nonempty"):
        filter_quality((), SFTQualityPolicy())
    with pytest.raises(TypeError, match="DollyRecord"):
        quality_rejection({}, SFTQualityPolicy())
