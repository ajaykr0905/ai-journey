import copy
import json

import pytest

from ai_journey.moe_route_audit import validate_moe_routing


def stats(*, applied=True):
    return {
        "layer_index": 0,
        "valid_tokens": 3,
        "capacity": 1 if applied else 6,
        "attempted_counts": (3, 2, 1, 0),
        "accepted_counts": (1, 1, 1, 0) if applied else (3, 2, 1, 0),
        "dropped_counts": (2, 1, 0, 0) if applied else (0, 0, 0, 0),
        "probability_sums": (0.75, 0.75, 0.75, 0.75),
        "capacity_applied": applied,
        "accepted_per_token": ((2, 0), (1, 0)) if applied else ((2, 2), (2, 0)),
        "token_indices": (0, 1, 2),
        "selected_experts": ((0, 1), (0, 1), (0, 2)),
        "accepted_routes": ((True, True), (False, False), (False, True))
        if applied
        else ((True, True),) * 3,
    }


def test_recompute_training_capacity_and_dropless_inference_without_mutation():
    for applied in (True, False):
        evidence = stats(applied=applied)
        saved = copy.deepcopy(evidence)
        result = validate_moe_routing(evidence, expert_count=4)
        assert evidence == saved
        assert result["attempted_routes"] == 6
        assert result["accepted_routes"] == (3 if applied else 6)
        assert result["dropped_routes"] == (3 if applied else 0)
        assert result["unrouted_tokens"] == (1 if applied else 0)
        assert result["partially_routed_tokens"] == (1 if applied else 0)
        assert result["fully_routed_tokens"] == (1 if applied else 3)
        assert (
            validate_moe_routing(json.loads(json.dumps(evidence)), expert_count=4)
            == result
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("layer_index", True),
        ("valid_tokens", 3.0),
        ("capacity", False),
        ("capacity_applied", 1),
        ("token_indices", (0, 0, 2)),
        ("token_indices", (0, 2, 9)),
        ("selected_experts", ((0, 0), (0, 1), (0, 2))),
        ("selected_experts", ((0, True), (0, 1), (0, 2))),
        ("selected_experts", ((0, 9), (0, 1), (0, 2))),
        ("accepted_routes", ((True, True), (False, False), (False, 1))),
        ("accepted_routes", ((True, False), (False, True), (False, True))),
        ("attempted_counts", (True, 2, 1, 0)),
        ("attempted_counts", (3, 1, 2, 0)),
        ("accepted_counts", (1, 1, 0, 1)),
        ("dropped_counts", (2, 0, 1, 0)),
        ("accepted_per_token", ((1, 1), (1, 0))),
        ("accepted_per_token", ((2, 0), (1, True))),
        ("probability_sums", (0.75, 0.75, float("nan"), 0.75)),
        ("probability_sums", (0.75, 0.75, float("inf"), 0.75)),
        ("probability_sums", (0.75, 0.75, True, 0.75)),
        ("probability_sums", (0.5, 0.5, 0.5, 0.5)),
        ("probability_sums", (4.0, -1.0, 0, 0)),
    ],
)
def test_reject_self_consistent_looking_but_impossible_or_mistyped_evidence(
    field, invalid
):
    evidence = stats()
    evidence[field] = invalid
    with pytest.raises(ValueError, match="routing"):
        validate_moe_routing(evidence, expert_count=4)


def test_zero_eligible_tokens_preserve_undefined_exposure_without_fake_activity():
    evidence = stats()
    evidence.update(
        valid_tokens=0,
        capacity=0,
        attempted_counts=(0,) * 4,
        accepted_counts=(0,) * 4,
        dropped_counts=(0,) * 4,
        probability_sums=(0.0,) * 4,
        accepted_per_token=((0, 0),),
        token_indices=(),
        selected_experts=(),
        accepted_routes=(),
    )
    result = validate_moe_routing(evidence, expert_count=4)
    assert all(
        result[name] == 0
        for name in (
            "valid_tokens",
            "attempted_routes",
            "accepted_routes",
            "dropped_routes",
        )
    )
    evidence["capacity"] = 1
    with pytest.raises(ValueError, match="empty-set"):
        validate_moe_routing(evidence, expert_count=4)


def test_schema_and_work_bounds_do_not_consume_arbitrary_iterators():
    evidence = stats()
    evidence["extra"] = "ignored?"
    with pytest.raises(ValueError, match="fields"):
        validate_moe_routing(evidence, expert_count=4)
    evidence = stats()
    evidence["valid_tokens"] = 65_537
    with pytest.raises(ValueError, match="valid_tokens"):
        validate_moe_routing(evidence, expert_count=4)
    evidence = stats()
    evidence["accepted_per_token"] = ((0,) * 4096,) * 64
    with pytest.raises(ValueError, match="cell budget"):
        validate_moe_routing(evidence, expert_count=4)
