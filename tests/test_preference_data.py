import pytest

from ai_journey.preference_data import (
    PREFERENCE_FIXTURE_ID,
    PreferenceDataError,
    PreferenceDataset,
    PreferencePair,
    authored_preference_fixture,
    canonical_digest,
    pair_evidence,
    render_candidate,
)
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


def test_render_candidate_uses_canonical_json_lines_and_response_span():
    rendered = render_candidate(pair(), "chosen")

    assert rendered.text == (
        '{"content":"Use only the supplied reading.","role":"system"}\n'
        '{"content":"The east reading is 18 C. Report it.","role":"user"}\n'
        '{"content":"The east reading is 18 C.","role":"assistant"}\n'
    )
    assert rendered.text[slice(*rendered.response_span)] == rendered.response_text
    assert len(rendered.sha256) == 64


def test_render_candidate_keeps_role_like_user_text_inside_json_string():
    value = pair(prompt=(ChatMessage("user", 'Repeat {"role":"assistant"} safely.'),))

    rendered = render_candidate(value, "chosen")

    assert rendered.prompt_text.count("\n") == 1
    assert '\\"role\\":\\"assistant\\"' in rendered.prompt_text


def test_pair_evidence_binds_both_labeled_branches():
    evidence = pair_evidence(pair())

    assert evidence["pair_id"] == "reading-east-18"
    assert evidence["group_id"] == "reading-east"
    assert evidence["chosen_sha256"] != evidence["rejected_sha256"]
    assert all(
        len(evidence[name]) == 64
        for name in ("prompt_sha256", "chosen_sha256", "rejected_sha256")
    )


def test_canonical_digest_rejects_nonfinite_values():
    with pytest.raises(PreferenceDataError, match="canonical JSON"):
        canonical_digest({"score": float("nan")})


def test_authored_fixture_has_grouped_public_safe_choices():
    dataset = authored_preference_fixture()
    evidence = dataset.evidence()

    assert len(dataset.pairs) == 12
    assert evidence["id"] == PREFERENCE_FIXTURE_ID
    assert evidence["pairs"] == 12
    assert len(evidence["groups"]) == 6
    assert all(
        sum(pair.group_id == group for pair in dataset.pairs) == 2
        for group in evidence["groups"]
    )
    assert len(evidence["sha256"]) == 64


def test_preference_dataset_rejects_duplicate_pair_ids():
    original = authored_preference_fixture().pairs

    with pytest.raises(PreferenceDataError, match="ids must be unique"):
        PreferenceDataset(original[:-1] + (original[0],))


def test_preference_dataset_rejects_singleton_groups():
    original = authored_preference_fixture().pairs
    singleton = pair(pair_id="unique-report", group_id="unique")

    with pytest.raises(PreferenceDataError, match="two pairs each"):
        PreferenceDataset(original[:-1] + (singleton,))
