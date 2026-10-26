"""Durable counters must be usable by the next deterministic CPU AdamW update."""

import copy

import pytest
import torch

from ai_journey.training_optimizer_state import validate_adamw_state


def initialized():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    return model, optimizer


@pytest.mark.parametrize("dtype", [torch.bool, torch.complex64, torch.complex128])
def test_unusable_counter_dtypes_fail_admission_without_mutation(dtype):
    model, optimizer = initialized()
    for state in optimizer.state.values():
        state["step"] = torch.tensor(1, dtype=dtype)
    before = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError, match="AdamW counter"):
        validate_adamw_state(model, optimizer, step=1)
    for key, state in optimizer.state_dict()["state"].items():
        assert state["step"].dtype == before["state"][key]["step"].dtype
        assert torch.equal(state["step"], before["state"][key]["step"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_admitted_real_cpu_counters_resume_an_actual_optimizer_update(dtype):
    model, optimizer = initialized()
    for state in optimizer.state.values():
        state["step"] = torch.tensor(1, dtype=dtype)
    validate_adamw_state(model, optimizer, step=1)
    optimizer.step()
    validate_adamw_state(model, optimizer, step=2)
    assert all(state["step"].item() == 2 for state in optimizer.state.values())
