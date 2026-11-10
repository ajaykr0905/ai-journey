from dataclasses import replace

import pytest

from ai_journey.sft_dataset_format import ChatTurn, format_dolly_record
from ai_journey.sft_dataset_identity import record_id, record_sha256
from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError


def record(context="Evidence paragraph."):
    return DollyRecord(
        instruction="Answer from the context.",
        context=context,
        response="Grounded answer.",
        category="closed_qa",
    )


def test_context_is_delimited_without_changing_the_instruction_or_answer():
    source = record()
    example = format_dolly_record(source)
    assert example.example_id == record_id(source)
    assert example.source_sha256 == record_sha256(source)
    assert example.messages == (
        ChatTurn("user", "Answer from the context.\n\nContext:\nEvidence paragraph."),
        ChatTurn("assistant", "Grounded answer."),
    )


def test_empty_context_does_not_add_a_synthetic_separator():
    example = format_dolly_record(record(context=""))
    assert example.messages[0].content == "Answer from the context."


def test_export_record_uses_a_standard_messages_array():
    example = format_dolly_record(record())
    assert example.as_record() == {
        "category": "closed_qa",
        "example_id": example.example_id,
        "messages": [
            {
                "content": "Answer from the context.\n\nContext:\nEvidence paragraph.",
                "role": "user",
            },
            {"content": "Grounded answer.", "role": "assistant"},
        ],
        "source_sha256": example.source_sha256,
    }


def test_chat_contract_rejects_role_order_and_identity_tampering():
    example = format_dolly_record(record())
    with pytest.raises(SFTDatasetError, match="one user turn"):
        replace(example, messages=(example.messages[1], example.messages[0]))
    with pytest.raises(SFTDatasetError, match="example_id"):
        replace(example, example_id="row-1")
    with pytest.raises(SFTDatasetError, match="source_sha256"):
        replace(example, source_sha256="0" * 63)


def test_formatter_requires_a_validated_source_record():
    with pytest.raises(TypeError, match="DollyRecord"):
        format_dolly_record({})
