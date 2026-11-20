from dataclasses import replace

import pytest

from ai_journey.matmul_checklist_source import MatmulChecklistError
from ai_journey.matmul_checklist_steps import (
    ALREADY_APPLIED,
    CHECKLIST,
    ITEM_SOURCE,
    ITEMS,
    MAX_TEXT,
    NEW_KERNEL,
    REFERENCE,
    checklist_mapping,
    checklist_sha256,
    require_checklist,
    step_of,
)
from ai_journey.matmul_checklist_variants import DAY_77, VARIANTS

# The digest of the plan as committed before its first timed run. A changed
# hypothesis, prediction or kernel name moves it, and a report seals it.
CHECKLIST_SHA256 = "cb5e3164e0d37716699dc9381b587faeeb40c2ca4f1a52d75b8ec586379323dc"


def test_the_plan_runs_s0_to_s7_and_applies_every_item():
    assert [step.step for step in CHECKLIST] == [f"S{index}" for index in range(8)]
    assert ITEMS == (
        "coalescing",
        "occupancy",
        "tiling",
        "arithmetic_intensity",
        "memory_vs_compute_bound",
        "control_divergence",
    )
    assert [step.item for step in CHECKLIST] == [
        "coalescing",
        "tiling",
        "arithmetic_intensity",
        "arithmetic_intensity",
        "memory_vs_compute_bound",
        "tiling",
        "occupancy",
        "control_divergence",
    ]
    assert "tracker notes" in ITEM_SOURCE
    assert REFERENCE == "matmul_x_columns"


def test_each_step_starts_from_the_kernel_that_the_step_before_it_keeps():
    assert CHECKLIST[0].base == "matmul_x_rows"
    for before, after in zip(CHECKLIST, CHECKLIST[1:]):
        assert after.base == before.chain
    six = CHECKLIST[6]
    assert six.variants == ("matmul_s6_group_8x8", "matmul_s6_group_32x32")
    assert six.chain == six.base == "matmul_s5_pad4"
    timed = [CHECKLIST[0].base, *(name for step in CHECKLIST for name in step.variants)]
    assert timed == ["matmul_x_rows", "matmul_x_columns", *list(VARIANTS)[2:]]


def test_only_s0_and_s1_reuse_earlier_kernels():
    for step in CHECKLIST:
        origins = {VARIANTS[name].origin for name in step.variants}
        if step.step in ("S0", "S1"):
            assert step.status == ALREADY_APPLIED and DAY_77 not in origins
        else:
            assert step.status == NEW_KERNEL and origins == {DAY_77}


def test_every_hypothesis_is_pinned_by_the_digest_of_the_plan():
    mapping = checklist_mapping()

    assert checklist_sha256() == CHECKLIST_SHA256
    assert mapping["steps"][4] == {
        "step": "S4",
        "item": "memory_vs_compute_bound",
        "status": NEW_KERNEL,
        "change": CHECKLIST[4].change,
        "hypothesis": CHECKLIST[4].hypothesis,
        "base": "matmul_s3_block_4x4",
        "predictions": [
            {"variant": "matmul_s4_vec4", "verdicts": ["no_clear_difference"]}
        ],
        "chain": "matmul_s4_vec4",
    }
    changed = list(CHECKLIST)
    changed[2] = replace(
        CHECKLIST[2], predictions=(("matmul_s2_coarse_4x1", ("slower",)),)
    )
    assert checklist_sha256(tuple(changed)) != CHECKLIST_SHA256
    for step in CHECKLIST:
        for text in (step.change, step.hypothesis):
            assert text.isascii() and len(text) <= MAX_TEXT
            assert ";" not in text


def test_the_predictions_name_each_new_kernel_with_known_verdicts():
    predicted = {
        name: step.predicted(name) for step in CHECKLIST for name in step.variants
    }

    assert predicted == {
        "matmul_x_columns": ("faster",),
        "matmul_tiled_32": ("slower", "no_clear_difference"),
        "matmul_s2_coarse_4x1": ("faster",),
        "matmul_s3_block_4x4": ("faster",),
        "matmul_s4_vec4": ("no_clear_difference",),
        "matmul_s5_pad4": ("no_clear_difference", "slower"),
        "matmul_s6_group_8x8": ("slower",),
        "matmul_s6_group_32x32": ("no_clear_difference", "slower"),
        "matmul_s7_interior": ("faster",),
    }
    assert step_of("matmul_x_rows").step == step_of("matmul_x_columns").step == "S0"
    assert step_of("matmul_s6_group_32x32").step == "S6"
    with pytest.raises(MatmulChecklistError, match="belongs to no checklist step"):
        step_of("matmul_tiled_16")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"step": "S10"}, "S and one digit"),
        ({"item": "privatization"}, "item must be one of"),
        ({"status": "planned"}, "status must be one of"),
        ({"change": ""}, "change must be ASCII text"),
        ({"hypothesis": "x" * (MAX_TEXT + 1)}, "hypothesis must be ASCII text"),
        ({"hypothesis": "S2 is faster \u2192 S1"}, "hypothesis must be ASCII text"),
        ({"predictions": ()}, "at least one kernel"),
        ({"predictions": (("matmul_tiled_32", ("faster",)),)}, "not a new kernel"),
        ({"predictions": (("matmul_s2_coarse_4x1", ()),)}, "distinct verdicts"),
        (
            {"predictions": (("matmul_s2_coarse_4x1", ("faster", "faster")),)},
            "distinct verdicts",
        ),
        ({"predictions": (("matmul_s2_coarse_4x1", ("better",)),)}, "verdicts"),
        ({"chain": "matmul_s3_block_4x4"}, "keep one of its kernels"),
        ({"status": ALREADY_APPLIED}, "exactly when its status"),
    ],
)
def test_a_malformed_step_is_refused(changes, message):
    with pytest.raises(MatmulChecklistError, match=message):
        replace(CHECKLIST[2], **changes)


def test_a_plan_that_breaks_the_chain_or_skips_an_item_is_refused():
    steps = list(CHECKLIST)

    with pytest.raises(MatmulChecklistError, match="S0, S1, ... in order"):
        require_checklist(tuple(reversed(steps)))
    with pytest.raises(MatmulChecklistError, match="must start from matmul_s3"):
        require_checklist((*steps[:4], replace(steps[4], base="matmul_tiled_32")))
    with pytest.raises(MatmulChecklistError, match="only once"):
        doubled = replace(
            steps[7],
            predictions=(("matmul_s4_vec4", ("faster",)),),
            chain="matmul_s5_pad4",
        )
        require_checklist((*steps[:7], doubled))
    with pytest.raises(MatmulChecklistError, match="every registered kernel"):
        require_checklist(tuple(steps[:7]))
    with pytest.raises(MatmulChecklistError, match="no step for occupancy"):
        missing = replace(steps[6], item="tiling")
        require_checklist((*steps[:6], missing, steps[7]))
    with pytest.raises(MatmulChecklistError, match="S0 must start"):
        require_checklist((replace(steps[0], base="matmul_tiled_32"), *steps[1:]))
    with pytest.raises(TypeError, match="ChecklistStep values"):
        require_checklist((steps[0].to_mapping(),))
