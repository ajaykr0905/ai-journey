import pytest

from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError, normalized_text


def record(**changes):
    values = {
        "instruction": "Answer this question.",
        "context": "Public context.",
        "response": "A grounded answer.",
        "category": "closed_qa",
    }
    values.update(changes)
    return DollyRecord(**values)


def test_record_normalizes_newlines_whitespace_and_unicode():
    value = record(instruction="  Cafe\u0301\r\nquestion  \r\n")
    assert value.instruction == "Caf\u00e9\nquestion"


def test_record_accepts_empty_context_but_not_empty_targets():
    assert record(context="  ").context == ""
    for field in ("instruction", "response"):
        with pytest.raises(SFTDatasetError, match="must not be empty"):
            record(**{field: "\n"})


def test_record_rejects_unknown_fields_categories_controls_and_oversize_text():
    with pytest.raises(SFTDatasetError, match="exactly four"):
        DollyRecord.from_mapping({"instruction": "x"})
    with pytest.raises(SFTDatasetError, match="category"):
        record(category="unknown")
    with pytest.raises(SFTDatasetError, match="control"):
        normalized_text("bad\x00text", "value")
    with pytest.raises(SFTDatasetError, match="byte limit"):
        record(response="x" * (64 * 1024 + 1))


def test_mapping_constructor_preserves_the_strict_schema():
    value = record()
    assert DollyRecord.from_mapping(value.__dict__) == value
