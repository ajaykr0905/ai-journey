import dataclasses

import pytest

from ai_journey.metal_liger_plan import (
    PLAN,
    TORCH_SEQUENCES,
    TORCH_SIDES,
    plan_mapping,
    plan_sha256,
    require_plan,
    step_of,
)
from ai_journey.metal_liger_source import FAMILIES, KERNELS, SWIGLU, MetalLigerError

# The digest that config/day-80-metal-liger.json pins. A change to any
# hypothesis, prediction or kernel text changes it.
DECLARED_PLAN_SHA256 = (
    "43d5f63eb9d93415d0b95244f7150d0652aa73a7845d98da7658993ff3f13165"
)


def test_the_plan_holds_one_step_per_family_with_its_kernel_digest():
    mapping = plan_mapping()

    assert [step["family"] for step in mapping["steps"]] == list(FAMILIES)
    for step in mapping["steps"]:
        assert step["source_sha256"] == KERNELS[step["kernel"]].source_sha256
        assert step["base"] == TORCH_SIDES[step["family"]]
        assert step["base_sequence"] == TORCH_SEQUENCES[step["family"]]
        assert step["predicted"]
    assert plan_sha256() == DECLARED_PLAN_SHA256
    assert step_of(SWIGLU).kernel == "liger_swiglu"


def test_a_changed_hypothesis_changes_the_plan_digest():
    changed = (dataclasses.replace(PLAN[0], hypothesis="RMSNorm is slower."),) + PLAN[
        1:
    ]

    assert plan_sha256(changed) != plan_sha256()


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ((), "tuple of steps"),
        (PLAN[:2], "every family once"),
        ((PLAN[1], PLAN[0], PLAN[2]), "every family once"),
    ],
)
def test_the_plan_must_cover_every_family_in_order(steps, message):
    with pytest.raises(MetalLigerError, match=message):
        require_plan(steps)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"family": "layernorm"}, "kernel family"),
        ({"hypothesis": ""}, "ASCII text"),
        ({"change": "x" * 601}, "ASCII text"),
        ({"predicted": ()}, "distinct verdicts"),
        ({"predicted": ("faster", "faster")}, "distinct verdicts"),
        ({"predicted": ("quicker",)}, "distinct verdicts"),
        ({"predicted": ["faster"]}, "distinct verdicts"),
    ],
)
def test_a_step_refuses_an_invalid_field(changes, message):
    with pytest.raises(MetalLigerError, match=message):
        dataclasses.replace(PLAN[0], **changes)
