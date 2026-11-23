import dataclasses

import pytest

from ai_journey.gpu_reduce_plan import (
    CLASS_NAMES,
    LARGE,
    MEDIUM,
    PLAN,
    SMALL,
    TORCH_SUM,
    ReduceStep,
    plan_mapping,
    plan_sha256,
    require_plan,
    size_class,
    step_of,
)
from ai_journey.gpu_reduce_source import KERNELS, R2, R3, STEP_KERNELS, GpuReduceError

# The digest of the declared plan, every hypothesis and kernel digest included.
PLAN_SHA256 = "b2134229a87d220e40772e732ca3d3e5d6c6a6ae82a90a475eb0405d87a7217f"


def test_the_plan_is_one_chain_from_torch_sum_over_every_step_kernel():
    assert [step.step for step in PLAN] == [f"R{index}" for index in range(6)]
    assert [step.kernel for step in PLAN] == list(STEP_KERNELS)
    assert [step.base for step in PLAN] == [TORCH_SUM, *STEP_KERNELS[:-1]]
    assert [step.same_order_as for step in PLAN] == [None, None, None, R2, None, None]
    assert plan_sha256() == PLAN_SHA256
    mapping = plan_mapping()
    assert (
        mapping["vendor"] == TORCH_SUM
        and mapping["reference_kernel"] == "reduce_r0_interleaved"
    )
    assert mapping["size_classes"] == [
        {"name": "small", "first_n": 1, "last_n": 262143},
        {"name": "medium", "first_n": 262144, "last_n": 4194303},
        {"name": "large", "first_n": 4194304, "last_n": 67108864},
    ]
    for step, entry in zip(PLAN, mapping["steps"]):
        assert entry["source_sha256"] == KERNELS[step.kernel].source_sha256
        assert [item["size_class"] for item in entry["predictions"]] == list(
            CLASS_NAMES
        )
    assert step_of(R3) is PLAN[3]
    with pytest.raises(GpuReduceError, match="belongs to no reduction step"):
        step_of("reduce_combine")


def test_each_size_class_has_the_declared_bounds():
    assert [size_class(n) for n in (1, 65536, 262143)] == [SMALL] * 3
    assert [size_class(n) for n in (262144, 1 << 20, (1 << 22) - 1)] == [MEDIUM] * 3
    assert [size_class(n) for n in (1 << 22, 10000019, 1 << 26)] == [LARGE] * 3
    for n in (0, (1 << 26) + 1, True, 2.0):
        with pytest.raises(GpuReduceError, match="outside 1 to"):
            size_class(n)
    assert PLAN[0].predicted(1 << 16) == ("faster", "no_clear_difference")
    assert PLAN[0].predicted(1 << 26) == ("slower",)
    assert PLAN[1].predicted(1 << 20) == ("faster",)
    assert PLAN[5].predicted(1 << 24) == ("no_clear_difference",)


def test_a_change_to_any_hypothesis_moves_the_digest():
    changed = (
        dataclasses.replace(PLAN[0], hypothesis=PLAN[0].hypothesis + " Again."),
        *PLAN[1:],
    )

    assert plan_sha256(changed) != PLAN_SHA256


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"step": "S0"}, "R and one digit"),
        ({"kernel": "reduce_r1_sequential"}, "its own registered kernel"),
        ({"change": ""}, "change must be ASCII text"),
        ({"hypothesis": "café"}, "hypothesis must be ASCII text"),
        ({"base": "reduce_other"}, "base 'reduce_other' is unknown"),
        ({"base": "reduce_r0_interleaved"}, "cannot be its own base"),
        ({"predictions": (("small", ("faster",)),)}, "each size class once"),
        (
            {
                "predictions": (
                    ("small", ("faster", "faster")),
                    ("medium", ("slower",)),
                    ("large", ("slower",)),
                )
            },
            "distinct verdicts",
        ),
        ({"same_order_as": "reduce_r2_first_add"}, "an earlier kernel only"),
    ],
)
def test_a_step_with_a_wrong_field_is_refused(change, message):
    fields = {**vars(PLAN[0]), **change}

    with pytest.raises(GpuReduceError, match=message):
        ReduceStep(**fields)


def test_a_plan_out_of_order_or_off_its_chain_is_refused():
    with pytest.raises(GpuReduceError, match="R0 to R5 in order"):
        require_plan(PLAN[1:])
    with pytest.raises(GpuReduceError, match="tuple of ReduceStep"):
        require_plan(list(PLAN))
    off_chain = dataclasses.replace(PLAN[2], base="reduce_r0_interleaved")
    with pytest.raises(GpuReduceError, match="must start from reduce_r1_sequential"):
        require_plan((*PLAN[:2], off_chain, *PLAN[3:]))
    vendor = dataclasses.replace(PLAN[0], base="reduce_r1_sequential")
    with pytest.raises(GpuReduceError, match="R0 must start from torch.sum"):
        require_plan((vendor, *PLAN[1:]))
