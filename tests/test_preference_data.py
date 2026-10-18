import pytest

from ai_journey.preference_data import PreferenceDataError, PreferencePair
from ai_journey.sft_data import ChatMessage, ToolCall


def pair(**overrides):
    values = {
        "pair_id": "reading-east-18",
        "group_id": "reading-east",
        "prompt": (
            ChatMessage("system", "Use only the supplied reading."),
            ChatMessage("user", "The east reading is 18 C. Report it."),
        ),
        "chosen": ChatMessage("assistant", "The east reading is 18 C."),
        "rejected": ChatMessage("assistant", "The east reading is 28 C."),
    }
    values.update(overrides)
    return PreferencePair(**values)


def test_preference_pair_accepts_a_plain_assistant_choice():
    value = pair()

    assert value.pair_id == "reading-east-18"
    assert value.prompt[-1].role == "user"
    assert value.chosen.content != value.rejected.content


@pytest.mark.parametrize("field", ["pair_id", "group_id"])
def test_preference_pair_rejects_unbounded_identifiers(field):
    with pytest.raises(PreferenceDataError, match=field):
        pair(**{field: "Bad identifier"})


@pytest.mark.parametrize(
    "prompt",
    [
        (),
        (ChatMessage("assistant", "Unexpected."),),
        (ChatMessage("user", "Question?"), ChatMessage("user", "Again?")),
        (ChatMessage("user", "Question?"), ChatMessage("assistant", "Answer.")),
    ],
)
def test_preference_pair_rejects_invalid_prompt_state(prompt):
    with pytest.raises(PreferenceDataError, match="prompt|alternating|end"):
        pair(prompt=prompt)


def test_preference_pair_rejects_tool_calls_in_candidates():
    call = ToolCall("call_reading", "reading.lookup", "{}")

    with pytest.raises(PreferenceDataError, match="chosen"):
        pair(chosen=ChatMessage("assistant", "", tool_call=call))


def test_preference_pair_rejects_identical_candidates():
    response = ChatMessage("assistant", "The east reading is 18 C.")

    with pytest.raises(PreferenceDataError, match="must differ"):
        pair(chosen=response, rejected=response)
