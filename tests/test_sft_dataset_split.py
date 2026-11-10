from collections import Counter

import pytest

from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError
from ai_journey.sft_dataset_split import split_records


def record(category, index):
    return DollyRecord(f"Question {category} {index}", "", f"Answer {index}", category)


def records():
    return tuple(
        [record("open_qa", index) for index in range(12)]
        + [record("classification", index) for index in range(6)]
        + [record("summarization", index) for index in range(2)]
    )


def test_split_is_exact_stratified_and_content_disjoint():
    result = split_records(records(), validation_count=5, seed=67)
    assert len(result.train) == 15
    assert len(result.validation) == 5
    assert Counter(row.category for row in result.validation) == {
        "open_qa": 3,
        "classification": 1,
        "summarization": 1,
    }
    assert not set(result.train).intersection(result.validation)


def test_split_is_source_order_independent_and_seeded():
    rows = records()
    first = split_records(rows, validation_count=5, seed=67)
    assert first == split_records(tuple(reversed(rows)), validation_count=5, seed=67)
    assert (
        first.validation != split_records(rows, validation_count=5, seed=68).validation
    )


def test_split_rejects_duplicate_content():
    rows = records()
    with pytest.raises(SFTDatasetError, match="duplicate"):
        split_records(rows + (rows[0],), validation_count=5, seed=67)


@pytest.mark.parametrize("validation_count", [2, 20, True])
def test_split_rejects_counts_that_lose_categories_or_training_data(validation_count):
    with pytest.raises(SFTDatasetError, match="validation count"):
        split_records(records(), validation_count=validation_count, seed=67)


def test_split_rejects_invalid_records_and_seed():
    with pytest.raises(SFTDatasetError, match="nonempty Dolly"):
        split_records((), validation_count=1, seed=67)
    with pytest.raises(SFTDatasetError, match="seed"):
        split_records(records(), validation_count=5, seed=-1)
