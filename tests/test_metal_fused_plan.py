import dataclasses
from hashlib import sha256

import pytest

from ai_journey.matmul_tiled_comparison import VERDICTS
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.metal_fused_plan import (
    BASELINE,
    PLAN,
    TORCH_SOFTMAX,
    FusedStep,
    plan_mapping,
    plan_sha256,
    require_plan,
    step_of,
)
from ai_journey.metal_fused_source import (
    F1,
    F2,
    F3,
    STEP_VARIANTS,
    U0,
    VARIANTS,
    MetalFusedError,
)


def test_the_plan_is_one_chain_from_the_vendor_side_through_every_step():
    assert [step.variant for step in PLAN] == list(STEP_VARIANTS)
    assert [step.base for step in PLAN] == [TORCH_SOFTMAX, U0, F1, F2]
    assert BASELINE == U0
    assert [step.predicted for step in PLAN] == [
        ("slower",),
        ("faster",),
        ("faster", "no_clear_difference"),
        ("no_clear_difference", "slower"),
    ]
    assert [step.same_bits_as for step in PLAN] == [None, U0, None, None]
    assert all(set(step.predicted) <= set(VERDICTS) for step in PLAN)


def test_the_digest_binds_every_hypothesis_and_kernel_text():
    mapping = plan_mapping()

    assert plan_sha256() == sha256(canonical_json_bytes(mapping)).hexdigest()
    assert mapping["baseline"] == U0 and mapping["vendor"] == TORCH_SOFTMAX
    for entry, step in zip(mapping["steps"], PLAN):
        assert entry["source_sha256"] == VARIANTS[step.variant].source_sha256
        assert entry["hypothesis"] == step.hypothesis
    changed = (
        dataclasses.replace(PLAN[0], hypothesis="U0 is faster than torch."),
        *PLAN[1:],
    )
    assert plan_sha256(changed) != plan_sha256()


def test_each_hypothesis_is_short_ascii_and_names_its_prediction():
    for step in PLAN:
        assert step.hypothesis.isascii() and len(step.hypothesis) <= 600
        assert ";" not in step.hypothesis and ";" not in step.change
    assert "1.1 or more" in step_of(F1).hypothesis
    assert "same order as U0" in step_of(F1).hypothesis
    assert "precise::exp twice" in step_of(F3).hypothesis


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"step": "X1"}, "step id"),
        ({"step": "F1"}, "its own registered variant"),
        ({"change": ""}, "change must be ASCII"),
        ({"hypothesis": "café"}, "hypothesis must be ASCII"),
        ({"hypothesis": "x" * 601}, "hypothesis must be ASCII"),
        ({"base": "torch_matmul"}, "is unknown"),
        ({"base": F2}, "its own base"),
        ({"predicted": ()}, "distinct verdicts"),
        ({"predicted": ("faster", "faster")}, "distinct verdicts"),
        ({"predicted": ("quicker",)}, "distinct verdicts"),
        ({"same_bits_as": F3}, "earlier step"),
        ({"same_bits_as": F2}, "earlier step"),
    ],
)
def test_a_step_refuses_an_invalid_field(changes, message):
    fields = {**dataclasses.asdict(step_of(F2)), **changes}

    with pytest.raises(MetalFusedError, match=message):
        FusedStep(**fields)


def test_the_plan_check_refuses_a_broken_chain():
    with pytest.raises(MetalFusedError, match="every step once"):
        require_plan(PLAN[:3])
    with pytest.raises(MetalFusedError, match="every step once"):
        require_plan((PLAN[0], PLAN[2], PLAN[1], PLAN[3]))
    rebased = dataclasses.replace(PLAN[3], base=U0)
    with pytest.raises(MetalFusedError, match="step before it"):
        require_plan((*PLAN[:3], rebased))
    with pytest.raises(MetalFusedError, match="start from torch_softmax"):
        require_plan((dataclasses.replace(PLAN[0], base=F1), *PLAN[1:]))
    with pytest.raises(TypeError):
        require_plan((*PLAN[:3], "F3"))
    with pytest.raises(MetalFusedError, match="no plan step"):
        step_of("torch_softmax")
