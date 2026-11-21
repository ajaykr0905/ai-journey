"""Tests of the strict plan parser and of the pinned Day 79 plan."""

import json
from pathlib import Path

import pytest

from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.triton_matmul_config import (
    MAX_INTERPRETER_STEPS,
    PINNED_PLAN,
    TritonMatmulPlan,
    interpreter_steps,
    load_plan,
)
from ai_journey.triton_matmul_kernel import AUTOTUNE_CONFIGS, TritonMatmulError

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-79-triton-matmul.json"


def mapping(**changes):
    return {**PINNED_PLAN.to_mapping(), **changes}


def test_the_config_file_is_the_pinned_plan_of_the_day_74_shapes():
    day_74 = json.loads((ROOT / "config/day-74-matmul-metal.json").read_text())

    assert load_plan(PINNED) == PINNED_PLAN
    assert PINNED_PLAN.seed == 74
    assert PINNED_PLAN.configs == AUTOTUNE_CONFIGS
    assert [[s.m, s.n, s.k] for s in PINNED_PLAN.shapes] == (
        day_74["correctness"]["shapes"]
    )
    assert TritonMatmulPlan.from_mapping(PINNED_PLAN.to_mapping()) == PINNED_PLAN
    assert interpreter_steps(PINNED_PLAN.shapes, PINNED_PLAN.configs) == 35640
    assert 35640 <= MAX_INTERPRETER_STEPS == 131072


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        (mapping(extra=1), "fields mismatch"),
        ({key: value for key, value in mapping().items() if key != "seed"}, "seed"),
        (mapping(schema_version=2), "schema_version"),
        (mapping(schema_version=True), "schema_version"),
        (mapping(name="Day 79"), "slug"),
        (mapping(seed=True), "seed"),
        (mapping(seed=2**32), "seed"),
        (mapping(correctness={"shapes": [[1, 1]]}), "triple"),
        (mapping(correctness={"shapes": [[1, 1, 1.0]]}), "k must be an integer"),
        (mapping(correctness={"shapes": [[1, 1, 8193]]}), "k must be in"),
        (mapping(correctness={"shapes": []}), "1 to 16"),
        (mapping(correctness={"shapes": [[17, 33, 9]] * 2}), "repeat a shape"),
        (mapping(correctness={"shapes": [[17, 33, 9]], "seed": 1}), "fields mismatch"),
        (mapping(configs=["m16_n16_k16_g1_w1_s1"]), "unknown config"),
        (mapping(configs=[]), "at least one"),
        (mapping(configs=["m32_n64_k32_g8_w2_s5", "m64_n32_k32_g8_w2_s5"]), "order"),
        (mapping(configs=["m64_n32_k32_g8_w2_s5"] * 2), "no repeat"),
        (mapping(configs="m64_n32_k32_g8_w2_s5"), "JSON array"),
    ],
)
def test_a_plan_that_breaks_a_rule_is_refused(changed, message):
    with pytest.raises((TritonMatmulError, TypeError), match=message):
        TritonMatmulPlan.from_mapping(changed)


def test_each_config_must_meet_a_ragged_edge_on_each_axis():
    aligned = mapping(correctness={"shapes": [[128, 128, 128], [17, 33, 9]]})

    with pytest.raises(TritonMatmulError, match="short later K block"):
        TritonMatmulPlan.from_mapping(aligned)
    ragged = mapping(correctness={"shapes": [[17, 33, 9], [31, 47, 129]]})
    assert len(TritonMatmulPlan.from_mapping(ragged).shapes) == 2


def test_the_interpreter_work_of_a_plan_is_bounded():
    big = mapping(
        correctness={"shapes": [[8192, 8192, 8192], [31, 47, 129]]},
        configs=["m32_n64_k32_g8_w2_s5"],
    )

    with pytest.raises(TritonMatmulError, match="interpreter steps"):
        TritonMatmulPlan.from_mapping(big)
    assert interpreter_steps((MatmulShape(31, 47, 129),), AUTOTUNE_CONFIGS[:1]) == 10


def test_a_plan_file_is_read_strictly(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"seed": 74, "seed": 75}', encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(PINNED)

    with pytest.raises(TritonMatmulError, match="duplicate JSON key"):
        load_plan(duplicate)
    with pytest.raises(TritonMatmulError, match="non-symlink"):
        load_plan(link)
    with pytest.raises(TritonMatmulError, match="must be a JSON object"):
        TritonMatmulPlan.from_mapping([1])
