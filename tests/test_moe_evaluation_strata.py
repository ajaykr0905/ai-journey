import copy
import math

import pytest
import torch

from ai_journey.moe_evaluation_strata import summarize_moe_token_losses


def inputs():
    return (
        torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]],
                [[4.0, 3.0, 2.0, 1.0], [2.0, 3.0, 4.0, 5.0]],
            ],
            requires_grad=True,
        ),
        torch.tensor([[0, 1], [2, 3]]),
        torch.tensor([[0, 1], [2, 0]]),
        torch.tensor([[True, True], [True, False]]),
    )


def test_route_strata_conserve_target_exposure_and_individual_token_losses():
    logits, targets, counts, mask = inputs()
    actual = summarize_moe_token_losses(logits, targets, counts, mask)
    expected = -logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    assert actual["target_tokens"] == 3
    assert actual["nll_sum"] == pytest.approx(
        float(expected[mask].detach().double().sum())
    )
    assert sum(row["target_tokens"] for row in actual["strata"]) == 3
    assert math.fsum(row["nll_sum"] for row in actual["strata"]) == actual["nll_sum"]
    for index, row in enumerate(actual["strata"]):
        assert row["accepted_routes"] == index
        assert row["target_tokens"] == 1
        assert row["mean_nll"] == row["nll_sum"]
    assert "not a causal estimate" in actual["scope"]


def test_empty_strata_are_null_and_padded_logits_do_not_enter_metrics():
    logits, targets, _, mask = inputs()
    counts = torch.where(mask, 2, 0)
    before = summarize_moe_token_losses(logits, targets, counts, mask)
    logits = logits.detach().clone()
    logits[~mask] = 1000.0
    assert summarize_moe_token_losses(logits, targets, counts, mask) == before
    assert before["strata"][0]["mean_nll"] is None
    assert before["strata"][0]["nll_sum"] == 0.0


def test_diagnostic_preserves_gradients_rng_and_every_input():
    values = inputs()
    values[0].grad = torch.ones_like(values[0])
    saved = [value.detach().clone() for value in values]
    gradient = values[0].grad.clone()
    rng = torch.get_rng_state().clone()
    summarize_moe_token_losses(*values)
    assert all(torch.equal(value, old) for value, old in zip(values, saved))
    assert torch.equal(values[0].grad, gradient)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize(
    "case",
    ["nan", "negative", "overfull", "padding", "mask", "target", "dtype", "shape"],
)
def test_rejects_invalid_numerics_alignment_and_route_accounting(case):
    values = list(inputs())
    values[0] = values[0].detach()
    if case == "nan":
        values[0][0, 0, 0] = float("nan")
    elif case == "negative":
        values[2][0, 0] = -1
    elif case == "overfull":
        values[2][0, 0] = 3
    elif case == "padding":
        values[2][1, 1] = 1
    elif case == "mask":
        values[3].fill_(False)
    elif case == "target":
        values[1][0, 0] = 4
    elif case == "dtype":
        values[2] = values[2].bool()
    else:
        values[2] = values[2][:1]
    with pytest.raises(ValueError):
        summarize_moe_token_losses(*values)


def test_joint_permutation_changes_no_loss_stratum():
    values = inputs()
    assert summarize_moe_token_losses(*values) == summarize_moe_token_losses(
        *(value.flip(0) for value in values)
    )


def test_non_tensor_input_is_rejected():
    values = list(inputs())
    values[2] = copy.deepcopy(values[2].tolist())
    with pytest.raises(TypeError):
        summarize_moe_token_losses(*values)
