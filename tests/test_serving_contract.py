import pytest

from ai_journey.serving_contract import (
    EngineIdentity,
    ModelIdentity,
    ServingContractError,
    WorkloadShape,
)


def test_workload_shape_bounds_total_token_demand():
    workload = WorkloadShape(
        request_count=8,
        prompt_tokens_per_request=128,
        output_tokens_per_request=32,
        max_concurrency=4,
    )

    assert workload.total_requested_tokens == 1280


@pytest.mark.parametrize(
    "field,value,exception",
    [
        ("request_count", 0, ServingContractError),
        ("prompt_tokens_per_request", True, TypeError),
        ("output_tokens_per_request", -1, ServingContractError),
        ("max_concurrency", 9, ServingContractError),
    ],
)
def test_workload_shape_rejects_ambiguous_or_unbounded_controls(
    field, value, exception
):
    values = {
        "request_count": 8,
        "prompt_tokens_per_request": 128,
        "output_tokens_per_request": 32,
        "max_concurrency": 4,
    }
    values[field] = value

    with pytest.raises(exception):
        WorkloadShape(**values)


def test_engine_identity_pins_name_version_and_revision():
    identity = EngineIdentity("vllm", "0.12.0", "abcdef1")

    assert (identity.name, identity.version, identity.revision) == (
        "vllm",
        "0.12.0",
        "abcdef1",
    )


@pytest.mark.parametrize("value", ["", " leading", "trailing "])
def test_engine_identity_rejects_noncanonical_text(value):
    with pytest.raises(ServingContractError, match="canonical"):
        EngineIdentity("vllm", "0.12.0", value)


def test_model_identity_pins_model_and_tokenizer_revisions():
    identity = ModelIdentity("org/model", "model-sha", "tokenizer-sha", 7_000_000_000)

    assert identity.parameter_count == 7_000_000_000
    assert identity.revision == "model-sha"
    assert identity.tokenizer_revision == "tokenizer-sha"


def test_model_identity_rejects_unbounded_parameter_counts():
    with pytest.raises(ServingContractError, match="positive"):
        ModelIdentity("org/model", "model-sha", "tokenizer-sha", 0)
