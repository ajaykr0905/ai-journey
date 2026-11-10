import pytest

from ai_journey.sft_dataset_dedup import SFTDedupResult, deduplicate_records
from ai_journey.sft_dataset_identity import record_id
from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError


def record(index):
    return DollyRecord(f"Question {index}", "", f"Answer {index}", "open_qa")


def test_exact_duplicates_are_removed_with_reconciled_evidence():
    first, second = record(1), record(2)
    result = deduplicate_records((first, second, first, first))
    assert result.input_records == 4
    assert result.duplicate_records == 2
    assert result.duplicate_counts == ((record_id(first), 3),)
    assert set(result.records) == {first, second}


def test_output_is_order_independent_and_sorted_by_content_identity():
    rows = (record(1), record(2), record(3), record(2))
    forward = deduplicate_records(rows)
    reverse = deduplicate_records(tuple(reversed(rows)))
    assert forward == reverse
    assert [record_id(value) for value in forward.records] == sorted(
        record_id(value) for value in forward.records
    )


def test_similar_but_distinct_examples_are_not_removed():
    result = deduplicate_records((record(1), record("01")))
    assert len(result.records) == 2
    assert result.duplicate_records == 0


def test_deduplication_rejects_empty_or_unvalidated_inputs():
    for value in ((), (record(1), {})):
        with pytest.raises(SFTDatasetError, match="nonempty Dolly"):
            deduplicate_records(value)


def test_dedup_result_rejects_inconsistent_metrics():
    with pytest.raises(SFTDatasetError, match="reconcile"):
        SFTDedupResult((record(1),), 2, 0, ())
