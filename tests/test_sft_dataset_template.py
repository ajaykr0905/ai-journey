import json

import pytest

from ai_journey.sft_dataset_format import format_dolly_record
from ai_journey.sft_dataset_record import DollyRecord, SFTDatasetError
from ai_journey.sft_dataset_template import (
    RenderedChat,
    assistant_character_mask,
    render_chatml,
)


def example(instruction="Answer safely.", response="A useful answer."):
    return format_dolly_record(
        DollyRecord(
            instruction=instruction,
            context="",
            response=response,
            category="open_qa",
        )
    )


def test_chatml_marks_exactly_the_json_encoded_assistant_target():
    rendered = render_chatml(example(response='Use "quotes".'))
    start, end = rendered.assistant_spans[0]
    assert rendered.text[start:end] == json.dumps('Use "quotes".', ensure_ascii=False)
    mask = assistant_character_mask(rendered)
    assert sum(mask) == end - start
    assert not any(mask[:start])
    assert not any(mask[end:])


def test_role_marker_text_remains_escaped_content_not_a_new_turn():
    rendered = render_chatml(
        example(
            instruction="Explain <|assistant|> literally.", response="Literal marker."
        )
    )
    assert rendered.text.count("\n<|assistant|>\n") == 1
    assert '"Explain <|assistant|> literally."' in rendered.text


def test_rendering_is_deterministic_and_content_sensitive():
    first = render_chatml(example())
    assert first == render_chatml(example())
    assert first.sha256 != render_chatml(example(response="Different answer.")).sha256


def test_rendered_contract_rejects_invalid_terminators_and_spans():
    with pytest.raises(SFTDatasetError, match="terminator"):
        RenderedChat("dolly-id", "incomplete", ((0, 1),))
    with pytest.raises(SFTDatasetError, match="intervals"):
        RenderedChat("dolly-id", "x<|end|>\n", ((3, 2),))


def test_template_and_mask_require_validated_values():
    with pytest.raises(TypeError, match="ChatExample"):
        render_chatml({})
    with pytest.raises(TypeError, match="RenderedChat"):
        assistant_character_mask("bad")
