"""Tests of the strict Day 81 plan and the pinned plan file."""

import json
from pathlib import Path

import pytest

from ai_journey.triton_flash_config import (
    MAX_INTERPRETER_STEPS,
    MAX_STORED_VALUES,
    PINNED_PLAN,
    TritonFlashPlan,
    interpreter_steps,
    load_plan,
    stored_values,
)
from ai_journey.triton_flash_kernel import FLASH_CONFIGS, TritonFlashError
from ai_journey.triton_flash_reference import AttentionCase

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-81-triton-flash.json"


def write(tmp_path, mapping):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return path


def test_the_pinned_file_is_the_pinned_plan():
    plan = load_plan(PINNED)

    assert plan == PINNED_PLAN
    assert plan.to_mapping() == json.loads(PINNED.read_text(encoding="utf-8"))
    assert plan.configs == FLASH_CONFIGS
    assert interpreter_steps(plan.cases, plan.configs) == 506
    assert stored_values(plan.cases, plan.configs) == 193488
    assert {case.causal for case in plan.cases} == {True, False}


@pytest.mark.parametrize(
    ("change", "message"),
    (
        (lambda plan: plan.update(extra=1), "fields mismatch"),
        (lambda plan: plan.pop("seed"), "fields mismatch"),
        (lambda plan: plan.update(schema_version=2), "schema_version"),
        (lambda plan: plan.update(name="Day 81"), "slug"),
        (lambda plan: plan.update(seed=True), "seed"),
        (lambda plan: plan.update(configs=["m32_n32_w2_s2"]), "unknown config"),
        (lambda plan: plan.update(configs=plan["configs"][::-1]), "declared order"),
        (lambda plan: plan["cases"][0].update(causal=0), "causal must be a bool"),
        (lambda plan: plan["cases"][0].update(window=4), "fields mismatch"),
        (lambda plan: plan["cases"].append(dict(plan["cases"][0])), "repeat"),
        (
            lambda plan: plan.update(cases=[c for c in plan["cases"] if c["causal"]]),
            "non-causal",
        ),
    ),
)
def test_a_plan_file_refuses_each_bad_field(tmp_path, change, message):
    mapping = json.loads(PINNED.read_text(encoding="utf-8"))
    change(mapping)

    with pytest.raises(TritonFlashError, match=message):
        load_plan(write(tmp_path, mapping))


def test_each_config_needs_a_short_last_query_block_and_key_block():
    cases = (AttentionCase(1, 64, 16, True), AttentionCase(1, 128, 16, False))

    with pytest.raises(TritonFlashError, match="short last query block"):
        TritonFlashPlan("even", 1, cases, (FLASH_CONFIGS[1],))


def test_the_plan_bounds_the_interpreter_work_and_the_stored_values():
    heavy = (
        AttentionCase(16, 1000, 128, True),
        AttentionCase(16, 999, 128, False),
    )
    with pytest.raises(TritonFlashError, match="interpreter steps"):
        TritonFlashPlan("heavy", 1, heavy, (FLASH_CONFIGS[3],))
    wide = (
        AttentionCase(16, 300, 128, True),
        AttentionCase(16, 301, 128, False),
    )
    assert interpreter_steps(wide, (FLASH_CONFIGS[0],)) <= MAX_INTERPRETER_STEPS
    assert stored_values(wide, (FLASH_CONFIGS[0],)) > MAX_STORED_VALUES
    with pytest.raises(TritonFlashError, match="output values"):
        TritonFlashPlan("wide", 1, wide, (FLASH_CONFIGS[0],))
