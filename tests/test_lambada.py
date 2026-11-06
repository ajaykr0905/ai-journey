from hashlib import sha256

import pytest

from ai_journey.lambada import LambadaError, LambadaExample, LambadaSource


def source_payload(sample=b"bounded public response"):
    return {
        "repository": "EleutherAI/lambada_openai",
        "revision": "900124bf3b8235c6daf21033af9948b3f07346c4",
        "dataset_server_url": "https://datasets-server.huggingface.co/first-rows",
        "sample_sha256": sha256(sample).hexdigest(),
        "sample_bytes": len(sample),
        "test_rows": 5153,
        "source_file_sha256": (
            "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226"
        ),
        "license": "MIT",
    }


def test_source_contract_pins_public_snapshot():
    sample = b"bounded public response"
    source = LambadaSource.from_mapping(source_payload(sample))
    source.verify_sample(sample)
    assert source.test_rows == 5153


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "other/dataset"),
        ("revision", "not-a-digest"),
        ("sample_bytes", True),
        ("test_rows", 0),
        ("license", "unknown"),
    ],
)
def test_source_contract_rejects_unpinned_metadata(field, value):
    payload = source_payload()
    payload[field] = value
    with pytest.raises(LambadaError):
        LambadaSource.from_mapping(payload)


def test_source_contract_rejects_changed_sample_bytes():
    source = LambadaSource.from_mapping(source_payload())
    with pytest.raises(LambadaError, match="snapshot"):
        source.verify_sample(b"changed")


def test_example_has_stable_content_identity():
    first = LambadaExample(7, "A short public-safe fixture ends here")
    second = LambadaExample(7, "A short public-safe fixture ends here")
    assert first.digest == second.digest
    assert len(first.digest) == 64


@pytest.mark.parametrize(
    ("row_index", "text"),
    [(-1, "valid text"), (True, "valid text"), (0, ""), (0, "contains\x00null")],
)
def test_example_rejects_invalid_or_unsafe_values(row_index, text):
    with pytest.raises(LambadaError):
        LambadaExample(row_index, text)
