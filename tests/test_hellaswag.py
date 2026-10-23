import copy

import pytest

from ai_journey.hellaswag import HellaSwagError, HellaSwagExample


def example_payload(index=24, label="3"):
    return {
        "ind": index,
        "activity_label": "Roof shingle removal",
        "ctx_a": "A man is sitting on a roof.",
        "ctx_b": "he",
        "ctx": "A man is sitting on a roof. he",
        "endings": [
            "is using wrap to wrap a pair of skis.",
            "is ripping level tiles off.",
            "is holding a rubik's cube.",
            "starts pulling up roofing on a roof.",
        ],
        "source_id": "activitynet~public-example",
        "split": "val",
        "split_type": "indomain",
        "label": label,
    }


def test_example_schema_normalizes_label_and_has_stable_digest():
    first = HellaSwagExample.from_mapping(example_payload())
    second = HellaSwagExample.from_mapping(example_payload(label=3))
    assert first == second
    assert first.label == 3
    assert len(first.digest) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ind", True),
        ("activity_label", ""),
        ("endings", ["one", "two", "three"]),
        ("endings", ["same"] * 4),
        ("label", "4"),
    ],
)
def test_example_schema_rejects_malformed_records(field, value):
    payload = copy.deepcopy(example_payload())
    payload[field] = value
    with pytest.raises(HellaSwagError):
        HellaSwagExample.from_mapping(payload)


def test_example_schema_rejects_unknown_fields():
    payload = example_payload()
    payload["unexpected"] = "field"
    with pytest.raises(HellaSwagError, match="fields"):
        HellaSwagExample.from_mapping(payload)
