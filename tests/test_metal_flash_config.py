import copy
import dataclasses
import json
from pathlib import Path

import pytest

from ai_journey.metal_flash_config import (
    FlashCase,
    compulsory_bytes,
    config_from_mapping,
    load_config,
)
from ai_journey.metal_flash_plan import (
    PLAN,
    SIDES,
    TORCH_SDPA,
    FlashStep,
    kernel_digests,
    plan_mapping,
    plan_sha256,
    require_plan,
)
from ai_journey.metal_flash_reference import dense_layout
from ai_journey.metal_flash_source import CAUSAL, F1, FULL, KERNELS, N0, MetalFlashError

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "day-81-metal-flash.json"


def day81():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def changed(path, value):
    payload = day81()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return payload


def test_the_plan_compares_the_three_sides_with_sealed_kernel_digests():
    assert SIDES == (N0, F1, TORCH_SDPA)
    assert [(step.first, step.base, step.predicted) for step in PLAN] == [
        (N0, TORCH_SDPA, ("slower",)),
        (F1, N0, ("faster",)),
        (F1, TORCH_SDPA, ("slower",)),
    ]
    mapping = plan_mapping()
    assert mapping["kernels"] == kernel_digests()
    assert set(kernel_digests()) == set(KERNELS) - {"flash_probe_exp"}
    assert "scratch prototype" in PLAN[2].hypothesis
    assert len(plan_sha256()) == 64


def test_a_changed_hypothesis_changes_the_plan_digest():
    other = (dataclasses.replace(PLAN[0], predicted=("faster",)), *PLAN[1:])

    assert plan_sha256(other) != plan_sha256()


@pytest.mark.parametrize(
    "change",
    [
        {"first": "f2"},
        {"base": N0, "first": N0},
        {"predicted": ()},
        {"predicted": ("faster", "faster")},
        {"hypothesis": ""},
        {"change": "x" * 701},
    ],
)
def test_a_malformed_step_is_refused(change):
    with pytest.raises(MetalFlashError):
        dataclasses.replace(PLAN[0], **change)


def test_a_plan_needs_distinct_steps():
    with pytest.raises(MetalFlashError, match="own name"):
        require_plan((PLAN[0], PLAN[0]))
    with pytest.raises(TypeError):
        require_plan((PLAN[0], "step"))
    assert isinstance(PLAN[0], FlashStep)


def test_the_day_81_config_loads_and_names_the_declared_plan():
    config = load_config(CONFIG)

    assert config.plan_sha256 == plan_sha256()
    assert [shape.length for shape in config.benchmark.shapes] == [
        512,
        1024,
        2048,
        4096,
        8192,
    ]
    assert {shape.sequences * shape.length for shape in config.benchmark.shapes} == {
        32768
    }
    assert config.benchmark.modes == (FULL, CAUSAL)
    points = config.benchmark.points()
    assert [(point.index, point.mode) for point in points[:3]] == [
        (0, FULL),
        (1, CAUSAL),
        (2, FULL),
    ]
    assert config.comparison.min_faster_ratio == 1.1
    work = config.planned_work()
    assert work["benchmark_points"] == 10
    assert work["timed_samples"] == 3 * 10 * 12
    assert config.to_mapping() == day81()
    assert config_from_mapping(config.to_mapping()).fingerprint() == (
        config.fingerprint()
    )


def test_compulsory_bytes_count_q_k_v_and_o_once():
    assert compulsory_bytes(dense_layout(2, 10)) == 16 * 64 * 20


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("plan_sha256",), "0" * 64, "declared plan"),
        (("benchmark", "repeats"), 10, "multiple of the 3"),
        (("benchmark", "modes"), ["causal", "full"], "order"),
        (("benchmark", "modes"), ["full", "full"], "distinct"),
        (("benchmark", "check_rows"), 1, "check_rows"),
        (("benchmark", "shapes"), [], "1 to 6"),
        (("correctness", "repeats"), 1, "repeats"),
        (
            ("correctness", "cases"),
            [{"lengths": [1, 31, 33], "inputs": "normal"}],
            "1000",
        ),
        (("name",), "Day 81", "slug"),
        (("schema_version",), 2, "schema_version"),
    ],
)
def test_a_config_that_breaks_a_rule_is_refused(path, value, match):
    with pytest.raises(MetalFlashError, match=match):
        config_from_mapping(changed(path, value))


@pytest.mark.parametrize(
    ("cases", "match"),
    [
        ([{"lengths": [31, 33, 1000], "inputs": "wide_scores"}], "1 token"),
        ([{"lengths": [1, 33, 1000], "inputs": "wide_scores"}], "shorter"),
        ([{"lengths": [1, 31, 64, 1024], "inputs": "wide_scores"}], "multiple of 32"),
        ([{"lengths": [1, 31, 33, 1000], "inputs": "normal"}], "wide_scores"),
        ([{"sequences": 2, "length": 1000, "inputs": "wide_scores"}], "1 token"),
    ],
)
def test_the_cases_must_cover_the_edges_of_the_kernels(cases, match):
    with pytest.raises(MetalFlashError, match=match):
        config_from_mapping(changed(("correctness", "cases"), cases))


def test_a_dense_case_keeps_its_form_and_a_ragged_case_needs_two_lengths():
    payload = day81()
    dense = config_from_mapping(payload).correctness.cases[3]
    assert dense.to_mapping() == {"sequences": 2, "length": 1024, "inputs": "normal"}
    with pytest.raises(MetalFlashError, match="ragged"):
        FlashCase((4, 4), "normal")
    with pytest.raises(MetalFlashError, match="fields"):
        config_from_mapping(changed(("benchmark", "shapes"), [{"length": 512}]))


def test_a_config_past_its_work_bound_is_refused():
    payload = copy.deepcopy(day81())
    payload["benchmark"]["target_bytes_per_sample"] = 1 << 34

    with pytest.raises(MetalFlashError, match="target_bytes_per_sample|side calls"):
        config_from_mapping(payload)
