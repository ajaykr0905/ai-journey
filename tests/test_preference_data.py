import random

import pytest

from ai_journey.preference_data import (
    PREFERENCE_FIXTURE_ID,
    PreferenceDataError,
    PreferenceDataset,
    PreferenceFold,
    PreferencePair,
    authored_preference_fixture,
    candidate_targets,
    canonical_digest,
    fold_evidence,
    grouped_preference_folds,
    pair_evidence,
    preference_codec,
    preference_schedule_evidence,
    preference_training_batches,
    render_candidate,
    target_evidence,
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


def test_grouped_folds_are_deterministic_complete_and_leak_free():
    dataset = authored_preference_fixture()

    folds = grouped_preference_folds(dataset, fold_count=3, seed=44)

    assert folds == grouped_preference_folds(dataset, fold_count=3, seed=44)
    assert len(folds) == 3
    assert sorted(pair.pair_id for fold in folds for pair in fold.heldout) == sorted(
        pair.pair_id for pair in dataset.pairs
    )
    for fold in folds:
        assert {pair.group_id for pair in fold.train}.isdisjoint(
            pair.group_id for pair in fold.heldout
        )
        assert len(fold.train) == 8
        assert len(fold.heldout) == 4
    assert len(fold_evidence(folds)["sha256"]) == 64


def test_grouped_folds_change_assignment_with_seed():
    dataset = authored_preference_fixture()
    first = grouped_preference_folds(dataset, fold_count=3, seed=44)
    second = grouped_preference_folds(dataset, fold_count=3, seed=45)

    assert fold_evidence(first)["sha256"] != fold_evidence(second)["sha256"]


@pytest.mark.parametrize(("fold_count", "seed"), [(1, 44), (9, 44), (3, -1)])
def test_grouped_folds_reject_invalid_controls(fold_count, seed):
    with pytest.raises(PreferenceDataError):
        grouped_preference_folds(
            authored_preference_fixture(), fold_count=fold_count, seed=seed
        )


def test_preference_fold_rejects_group_leakage():
    pairs = authored_preference_fixture().pairs

    with pytest.raises(PreferenceDataError, match="disjoint"):
        PreferenceFold(0, pairs[:4], pairs[1:3])


def test_candidate_targets_score_every_response_character_once():
    dataset = authored_preference_fixture()
    value = dataset.pairs[0]
    codec = preference_codec(dataset)
    rendered = render_candidate(value, "chosen")

    targets = candidate_targets(value, "chosen", codec, block_size=16)

    assert targets.target_tokens == len(rendered.response_text)
    assert codec.decode(targets.targets) == rendered.response_text
    assert all(1 <= len(context) <= 16 for context in targets.contexts)
    assert targets.contexts[-1] == codec.encode(rendered.text)[-17:-1]


def test_preference_codec_can_include_warmup_text():
    dataset = authored_preference_fixture()

    codec = preference_codec(dataset, extra_text="₹")

    assert codec.decode(codec.encode("₹")) == "₹"


def test_target_evidence_binds_both_labels_and_context_policy():
    dataset = authored_preference_fixture()
    evidence = target_evidence(dataset.pairs, preference_codec(dataset), block_size=32)

    assert len(evidence["rows"]) == len(dataset.pairs) * 2
    assert evidence["target_tokens"] == sum(
        row["target_tokens"] for row in evidence["rows"]
    )
    assert len(evidence["sha256"]) == 64


def test_candidate_targets_rejects_invalid_block_size():
    dataset = authored_preference_fixture()

    with pytest.raises(PreferenceDataError, match="block_size"):
        candidate_targets(
            dataset.pairs[0], "chosen", preference_codec(dataset), block_size=0
        )


def test_preference_schedule_is_deterministic_and_rng_isolated():
    pairs = authored_preference_fixture().pairs
    random.seed(901)
    before = random.getstate()

    first = preference_training_batches(pairs, batch_size=3, steps=4, seed=44)
    second = preference_training_batches(pairs, batch_size=3, steps=4, seed=44)

    assert first == second
    assert random.getstate() == before
    evidence = preference_schedule_evidence(first)
    assert evidence["steps"] == 4
    assert evidence["batch_size"] == 3
    assert evidence["comparisons"] == 12
    assert len(evidence["sha256"]) == 64


def test_preference_schedule_changes_with_seed():
    pairs = authored_preference_fixture().pairs
    first = preference_training_batches(pairs, batch_size=3, steps=4, seed=44)
    second = preference_training_batches(pairs, batch_size=3, steps=4, seed=45)

    assert preference_schedule_evidence(first) != preference_schedule_evidence(second)


@pytest.mark.parametrize(
    ("batch_size", "steps", "seed"), [(0, 2, 44), (2, 0, 44), (2, 2, -1)]
)
def test_preference_schedule_rejects_invalid_controls(batch_size, steps, seed):
    with pytest.raises(PreferenceDataError):
        preference_training_batches(
            authored_preference_fixture().pairs,
            batch_size=batch_size,
            steps=steps,
            seed=seed,
        )
