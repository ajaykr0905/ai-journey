import json
from pathlib import Path

import pytest

from ai_journey.serving_recovery_model import (
    BOUNDARY,
    GenerationRequest,
    build_serving_model,
    completion_body,
    generate_completion,
    model_from_mapping,
    parse_corpus,
    request_from_mapping,
)

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "day-19-demo-names.txt"
FIXTURE_MODEL_SHA256 = (
    "da40d8977bc4380138fef8d4aa968da669bfb3293afc8fa271a62efd00c34c22"
)


def fixture_model():
    return build_serving_model(FIXTURE.read_bytes())


def test_fixture_model_counts_every_boundary_delimited_bigram():
    model = fixture_model()
    words = parse_corpus(FIXTURE.read_bytes())

    assert model.vocabulary[0] == BOUNDARY
    assert model.vocabulary[1:] == tuple(sorted(set("".join(words))))
    assert model.row_totals[0] == len(words)
    assert sum(model.row_totals) == sum(len(word) + 1 for word in words)
    assert model.model_sha256 == FIXTURE_MODEL_SHA256


def test_model_identity_ignores_line_order_and_surrounding_space():
    shuffled = b"\n".join(
        b"  " + line + b"\r" for line in reversed(FIXTURE.read_bytes().splitlines())
    )

    assert build_serving_model(shuffled) == fixture_model()
    assert model_from_mapping(fixture_model().to_mapping()) == fixture_model()


def test_completion_is_a_pure_function_of_request_and_model_digest():
    model = fixture_model()
    requests = [GenerationRequest("a", seed, 12) for seed in range(16)]
    first = [completion_body(model, request) for request in requests]

    assert first == [completion_body(fixture_model(), r) for r in requests]
    assert len(set(first)) > 4
    for request, body in zip(requests, first):
        payload = json.loads(body)
        output = payload.pop("output")
        assert payload == request.to_mapping() | {
            "finish_reason": "length" if len(output) == 12 else "eos",
            "model_sha256": FIXTURE_MODEL_SHA256,
        }
        assert set(output) <= set(model.vocabulary[1:])
        assert (
            body
            == json.dumps(
                json.loads(body), sort_keys=True, separators=(",", ":")
            ).encode()
        )


def test_generation_only_conditions_on_the_final_prompt_letter_and_budget():
    model = fixture_model()

    output, reason = generate_completion(model, GenerationRequest("zar", 9, 3))
    assert (len(output), reason) == (3, "length")
    assert completion_body(model, GenerationRequest("a", 3, 12)) != completion_body(
        model, GenerationRequest("a", 3, 11)
    )


def test_changed_parameters_change_the_model_digest_and_outputs():
    model = fixture_model()
    mapping = model.to_mapping()
    mapping["counts"][1][2] += 1
    changed = model_from_mapping(mapping)

    assert changed.model_sha256 != model.model_sha256
    request = GenerationRequest("a", 3, 12)
    assert completion_body(changed, request) != completion_body(model, request)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (b"", "non-empty"),
        (b"alpha\n\nbeta\n", "line 2 is empty"),
        (b"Alpha\n", "lowercase ASCII"),
        (b"al.pha\n", "lowercase ASCII"),
        (b"caf\xc3\xa9\n", "lowercase ASCII"),
        (b"\xff\n", "UTF-8"),
        (b"a" * (64 * 1024 + 1), "64 KiB"),
    ],
)
def test_corpus_parser_rejects_ambiguous_sources(source, message):
    with pytest.raises(ValueError, match=message):
        build_serving_model(source)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: m | {"vocabulary": ["a", "."]}, "boundary"),
        (lambda m: m | {"vocabulary": [".", "b", "a"] + m["vocabulary"][3:]}, "sorted"),
        (lambda m: m | {"counts": m["counts"][:-1]}, "square"),
        (lambda m: m | {"counts": [[0] * 18] + m["counts"][1:]}, "transition"),
        (
            lambda m: m | {"counts": [[1] + m["counts"][0][1:]] + m["counts"][1:]},
            "empty",
        ),
        (lambda m: m | {"counts": [[True] * 18] + m["counts"][1:]}, "integer"),
        (lambda m: m | {"kind": "other"}, "schema"),
        (lambda m: m | {"extra": 1}, "fields"),
    ],
)
def test_model_mapping_rejects_invalid_parameters(change, message):
    with pytest.raises((TypeError, ValueError), match=message):
        model_from_mapping(change(fixture_model().to_mapping()))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"prompt": "a", "seed": 1}, "fields"),
        ({"prompt": "a", "seed": 1, "max_new_tokens": 2, "x": 1}, "fields"),
        ({"prompt": "", "seed": 1, "max_new_tokens": 2}, "prompt"),
        ({"prompt": "A", "seed": 1, "max_new_tokens": 2}, "prompt"),
        ({"prompt": "a" * 33, "seed": 1, "max_new_tokens": 2}, "prompt"),
        ({"prompt": "a", "seed": True, "max_new_tokens": 2}, "integer"),
        ({"prompt": "a", "seed": -1, "max_new_tokens": 2}, "seed"),
        ({"prompt": "a", "seed": 1, "max_new_tokens": 33}, "max_new_tokens"),
    ],
)
def test_generation_requests_reject_ambiguous_inputs(payload, message):
    with pytest.raises((TypeError, ValueError), match=message):
        request_from_mapping(payload)


def test_generation_rejects_letters_outside_the_vocabulary():
    with pytest.raises(ValueError, match="outside the vocabulary"):
        generate_completion(fixture_model(), GenerationRequest("abc", 1, 4))
