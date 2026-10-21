import pytest
import torch
from ai_journey.ddp_generation import SamplingConfig, filter_logits
from ai_journey.ddp_generation import process_logits


def test_nucleus_retains_the_probability_crossing_token():
    logits = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64).log()
    filtered = filter_logits(logits, SamplingConfig(top_p=0.6))
    assert torch.isfinite(filtered).tolist() == [[True, True, False]]


def test_top_k_ties_have_stable_exact_cardinality():
    filtered = filter_logits(torch.zeros(1, 4), SamplingConfig(top_k=2))
    assert torch.isfinite(filtered).tolist() == [[True, True, False, False]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": float("nan")},
        {"top_k": True},
        {"top_p": 0},
        {"max_new_tokens": 10000},
    ],
)
def test_invalid_sampling_budgets_fail(kwargs):
    with pytest.raises(ValueError):
        SamplingConfig(**kwargs)


def test_repetition_penalty_is_sign_aware_and_applied_once_per_unique_id():
    logits = torch.tensor([[4.0, -2.0, 1.0]])
    result = process_logits(
        logits, torch.tensor([[0, 0, 1]]), repetition_penalty=2, banned_tokens=[2]
    )
    assert result[0, :2].tolist() == [2.0, -4.0]
    assert torch.isneginf(result[0, 2])
    assert logits.tolist() == [[4, -2, 1]]
    with pytest.raises(ValueError, match="every possible"):
        process_logits(logits, torch.tensor([[0]]), banned_tokens=[0, 1, 2])
