import pytest
import torch

from ai_journey.ddp_generation import SamplingConfig, TokenSampler


@pytest.mark.parametrize("seed", [2**64, 2**1000])
def test_request_seed_overflow_is_rejected_when_validating_sampling(seed):
    with pytest.raises(ValueError, match="seed"):
        SamplingConfig(seed=seed)


@pytest.mark.parametrize("seed", [0, 2**63, 2**64 - 1])
def test_every_supported_unsigned_seed_initializes_request_local_rng(seed):
    caller = torch.get_rng_state().clone()
    first = TokenSampler(SamplingConfig(seed=seed))
    second = TokenSampler(SamplingConfig(seed=seed))
    assert torch.equal(first.state(), second.state())
    assert torch.equal(caller, torch.get_rng_state())


@pytest.mark.parametrize("temperature", [1e-320, 1e-100, 0.001, 1.0, 1e100, 1e308])
def test_temperature_scaling_never_overflows_positive_logits(temperature):
    from ai_journey.ddp_generation import filter_logits

    config = SamplingConfig(temperature=temperature)
    values = filter_logits(
        torch.tensor([[1.0, 2.0, 3.0], [-torch.inf, -2.0, -1.0]]), config
    )
    probabilities = values.softmax(-1)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(-1), torch.ones(2, dtype=torch.float64)
    )
    assert probabilities[:, -1].ge(probabilities[:, 0]).all()


def test_stable_temperature_scaling_preserves_softmax_shift_invariance():
    from ai_journey.ddp_generation import filter_logits

    values = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    config = SamplingConfig(temperature=0.5, top_p=0.9)
    expected = filter_logits(values, config).softmax(-1)
    actual = filter_logits(values + 1e6, config).softmax(-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
