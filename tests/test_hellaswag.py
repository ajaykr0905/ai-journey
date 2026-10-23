import copy

import pytest

from ai_journey.hellaswag import HellaSwagError, HellaSwagExample, HellaSwagSource


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
    assert first.canonical_context == "A man is sitting on a roof. he"
    assert len(first.digest) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ind", True),
        ("activity_label", ""),
        ("endings", ["one", "two", "three"]),
        ("endings", ["same"] * 4),
        ("label", "4"),
        ("ctx", "A different context"),
        ("split", "train"),
        ("split_type", "unknown"),
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


def source_payload():
    return {
        "repository": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "validation_url": "https://huggingface.co/example.parquet",
        "validation_sha256": "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd",
        "validation_bytes": 6315951,
        "validation_rows": 10042,
        "license": "MIT",
    }


def test_source_contract_pins_public_validation_bytes():
    source = HellaSwagSource.from_mapping(source_payload())
    assert source.validation_rows == 10042
    assert source.license == "MIT"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "main"),
        ("validation_sha256", "not-a-digest"),
        ("validation_bytes", True),
        ("validation_rows", 0),
        ("license", "unknown"),
    ],
)
def test_source_contract_rejects_unpinned_or_invalid_metadata(field, value):
    payload = source_payload()
    payload[field] = value
    with pytest.raises(HellaSwagError):
        HellaSwagSource.from_mapping(payload)
