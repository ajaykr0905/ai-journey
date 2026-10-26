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
