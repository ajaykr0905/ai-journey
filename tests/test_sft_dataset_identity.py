import math

import pytest

from ai_journey.sft_dataset_identity import (
    canonical_json_bytes,
    record_id,
    record_sha256,
)
from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError


def record(**changes):
    values = {
        "instruction": "Summarize the passage.",
        "context": "A short public passage.",
        "response": "A short summary.",
        "category": "summarization",
    }
    values.update(changes)
    return DollyRecord(**values)


def test_record_identity_is_stable_canonical_and_bounded():
    value = record()
    assert record_sha256(value) == record_sha256(value)
    assert record_id(value) == f"dolly-{record_sha256(value)[:24]}"
    assert len(record_sha256(value)) == 64


def test_each_semantic_source_field_changes_identity():
    baseline = record_sha256(record())
    assert baseline != record_sha256(record(instruction="Explain the passage."))
    assert baseline != record_sha256(record(context="Another passage."))
    assert baseline != record_sha256(record(response="Another summary."))
    assert baseline != record_sha256(record(category="open_qa"))


def test_canonical_json_orders_fields_and_terminates_one_record():
    assert canonical_json_bytes({"b": 2, "a": "é"}) == b'{"a":"\xc3\xa9","b":2}\n'


def test_canonical_json_rejects_nonfinite_or_unserializable_values():
    for value in ({"value": math.nan}, {"value": object()}):
        with pytest.raises(SFTDatasetError, match="canonical JSON"):
            canonical_json_bytes(value)


def test_record_identity_requires_a_validated_record():
    with pytest.raises(TypeError, match="DollyRecord"):
        record_sha256({})
