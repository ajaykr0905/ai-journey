"""Route identity, stable numerics, masks and gradients are observable contracts."""

import pytest
import torch

from ai_journey.moe_routing import Top2Router, top2_routes


def test_top2_weights_and_ties_have_explicit_order():
    logits = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 3.0, 2.0, -1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    routes = top2_routes(logits)
    assert routes.expert_indices.tolist() == [[0, 1], [1, 2]]
    torch.testing.assert_close(
        routes.weights.sum(-1), torch.ones(2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        routes.probabilities.sum(-1), torch.ones(2, dtype=torch.float64)
    )
    (routes.weights[:, 0].sum() + routes.probabilities[:, 0].sum()).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_extreme_finite_logits_have_finite_selected_weights():
    maximum = torch.finfo(torch.float32).max
    routes = top2_routes(torch.tensor([[maximum, -maximum, 0.0]]))
    assert routes.expert_indices.tolist() == [[0, 2]]
    assert torch.isfinite(routes.weights).all()
    assert routes.weights.sum() == 1


def test_padding_is_excluded_before_projection_and_does_not_poison_gradients():
    router = Top2Router(2, 4).double()
    values = torch.tensor(
        [[[1.0, 2.0], [float("nan"), float("nan")], [3.0, 4.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    routes = router(values, token_mask=torch.tensor([[True, False, True]]))
    assert routes.token_indices.tolist() == [0, 2]
    assert routes.probabilities.shape == (2, 4)
    routes.weights[:, 0].sum().backward()
    assert torch.isfinite(router.projection.weight.grad).all()
    assert values.grad[0, 1].tolist() == [0.0, 0.0]


def test_all_masked_router_is_well_formed():
    routes = Top2Router(2, 4)(
        torch.ones(1, 3, 2), token_mask=torch.zeros(1, 3, dtype=torch.bool)
    )
    assert routes.token_shape == (1, 3)
    assert routes.token_indices.numel() == 0
    assert routes.expert_indices.shape == (0, 2)
    assert routes.probabilities.shape == (0, 4)


@pytest.mark.parametrize(
    "logits,mask",
    [
        (torch.ones(2, 1), None),
        (torch.ones(2, 65), None),
        (torch.ones(2, 4, dtype=torch.long), None),
        (torch.tensor([[float("inf"), 0.0]]), None),
        (torch.ones(2, 4), torch.ones(2)),
        (torch.ones(2, 4), torch.ones(1, dtype=torch.bool)),
    ],
)
def test_invalid_routing_contracts_fail(logits, mask):
    with pytest.raises(ValueError):
        top2_routes(logits, token_mask=mask)


def test_selected_weight_gradient_matches_double_precision_reference():
    logits = torch.tensor(
        [[0.1, 0.7, -0.5, 0.2]], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(lambda x: top2_routes(x).weights, (logits,))


def test_router_budget_rejects_before_linear(monkeypatch):
    import ai_journey.moe_routing as routing

    router = Top2Router(2, 4)
    monkeypatch.setattr(routing, "MAX_ROUTER_ENTRIES", 3)
    monkeypatch.setattr(
        router.projection, "forward", lambda _: pytest.fail("projection allocated")
    )
    with pytest.raises(ValueError, match="budget"):
        router(torch.ones(1, 2))
