import pytest

from ai_journey.serving_contract import ServingContractError, WorkloadShape


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
