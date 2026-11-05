import pytest

from ai_journey.scaling_experiment_identity import ExperimentInputs
from ai_journey.scaling_laws import ScalingLawError


def inputs(**overrides):
    values = {
        "source_url": "https://example.org/public.txt",
        "source_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "code_revision": "c" * 40,
    }
    values.update(overrides)
    return ExperimentInputs(**values)


def test_experiment_inputs_bind_public_bytes_tokenizer_and_code():
    identity = inputs()

    assert identity.source_url.endswith("public.txt")
    assert identity.source_sha256 == "a" * 64
    assert identity.tokenizer_sha256 == "b" * 64
    assert identity.code_revision == "c" * 40


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("source_url", "file:///private.txt", "HTTP"),
        ("source_sha256", "A" * 64, "SHA-256"),
        ("tokenizer_sha256", "a" * 63, "SHA-256"),
        ("code_revision", "c" * 7, "full lowercase"),
    ],
)
def test_experiment_inputs_reject_unverifiable_identity_fields(field, value, message):
    with pytest.raises(ScalingLawError, match=message):
        inputs(**{field: value})
