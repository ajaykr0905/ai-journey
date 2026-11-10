import pytest

from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError
from ai_journey.sft_dataset_select import select_records


def record(category, index):
    return DollyRecord(f"Question {category} {index}", "", f"Answer {index}", category)


def candidates():
    return tuple(
        [record("open_qa", index) for index in range(10)]
        + [record("classification", index) for index in range(5)]
        + [record("summarization", index) for index in range(3)]
    )


def test_selection_returns_exact_seeded_category_quotas():
    result = select_records(candidates(), target=9, seed=67)
    assert len(result.records) == 9
    assert result.candidate_records == 18
    assert dict(result.quotas) == {
        "classification": 3,
        "open_qa": 5,
        "summarization": 1,
    }


def test_selection_is_independent_of_source_order():
    rows = candidates()
    assert select_records(rows, target=9, seed=67) == select_records(
        tuple(reversed(rows)), target=9, seed=67
    )


def test_seed_changes_membership_without_changing_quotas():
    first = select_records(candidates(), target=9, seed=1)
    second = select_records(candidates(), target=9, seed=2)
    assert first.records != second.records
    assert first.quotas == second.quotas


def test_selection_rejects_duplicates_invalid_seed_and_impossible_target():
    rows = candidates()
    with pytest.raises(SFTDatasetError, match="deduplicated"):
        select_records(rows + (rows[0],), target=9, seed=67)
    with pytest.raises(SFTDatasetError, match="seed"):
        select_records(rows, target=9, seed=True)
    with pytest.raises(SFTDatasetError, match="target"):
        select_records(rows, target=2, seed=67)


def test_selection_rejects_empty_or_unvalidated_candidates():
    for rows in ((), (record("open_qa", 1), {})):
        with pytest.raises(SFTDatasetError, match="nonempty Dolly"):
            select_records(rows, target=1, seed=67)
