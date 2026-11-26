import copy
import dataclasses
import json
from pathlib import Path

import pytest

from ai_journey.matmul_tiled_comparison import VERDICTS
from ai_journey.metal_tc_config import (
    MetalTcConfig,
    TcCase,
    calls_per_sample,
    config_from_mapping,
    load_config,
)
from ai_journey.metal_tc_plan import (
    PLAN,
    SIDE_PRECISION,
    SIDES,
    TORCH_SIDES,
    TcStep,
    kernel_digests,
    plan_mapping,
    plan_sha256,
    require_plan,
)
from ai_journey.metal_tc_source import KERNELS, VARIANTS, MetalTcError

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config" / "day-82-metal-tc.json"


def pinned():
    return json.loads(PINNED.read_text(encoding="utf-8"))


def test_the_plan_names_seven_sides_and_nine_steps_with_predictions():
    assert SIDES == (*VARIANTS, *TORCH_SIDES)
    assert [step.step for step in PLAN] == [
        "m1_over_s1",
        "m2_over_m1",
        "m3_over_m1",
        "m3_over_s1",
        "torch_float16_over_torch_float32",
        "torch_bfloat16_over_torch_float32",
        "m1_over_torch_float32",
        "m2_over_torch_float16",
        "m3_over_torch_bfloat16",
    ]
    for step in PLAN:
        assert set(step.predicted) <= set(VERDICTS)
        assert len(step.hypothesis) <= 700 and step.hypothesis.isascii()
    assert SIDE_PRECISION["torch_bfloat16"] == SIDE_PRECISION["m3"] == "bfloat16"


def test_the_plan_digest_binds_every_hypothesis_and_kernel_text():
    mapping = plan_mapping()
    changed = dataclasses.replace(PLAN[0], hypothesis="M1 is slower than S1.")

    assert mapping["kernels"] == kernel_digests()
    assert set(mapping["kernels"]) == {k for k in KERNELS if KERNELS[k].variant}
    assert plan_sha256((changed, *PLAN[1:])) != plan_sha256()
    with pytest.raises(MetalTcError):
        require_plan((PLAN[0], PLAN[0]))
    with pytest.raises(MetalTcError):
        TcStep("x", "m1", "m1", "change", "hypothesis", ("faster",))
    with pytest.raises(MetalTcError):
        TcStep("x", "m1", "s1", "change", "hypothesis", ("quicker",))


def test_the_pinned_config_names_the_plan_and_plans_its_work():
    config = load_config(PINNED)
    work = config.planned_work()

    assert config.plan_sha256 == plan_sha256()
    assert config.benchmark.sizes == (512, 1024, 2048, 4096)
    assert [calls_per_sample(s, 1 << 34) for s in config.benchmark.sizes] == [
        64,
        8,
        1,
        1,
    ]
    assert work["timed_samples"] == 7 * 4 * 21
    assert work["correctness_launches"] == 6 * 4 * 2
    assert config_from_mapping(config.to_mapping()) == config
    assert len(config.fingerprint()) == 64


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("plan_sha256",), "0" * 64, "declared plan"),
        (("benchmark", "repeats"), 20, "multiple of the 7"),
        (("benchmark", "sizes"), [1024, 512], "increasing order"),
        (("benchmark", "check_rows"), 1, "check_rows"),
        (("benchmark", "target_flops_per_sample"), 1, "target_flops"),
        (("correctness", "repeats"), 1, "repeats"),
        (("comparison", "min_faster_ratio"), 1.0, "min_faster_ratio"),
        (("name",), "Day 82", "slug"),
        (("schema_version",), 2, "schema_version"),
    ],
)
def test_a_config_outside_its_contract_is_refused(path, value, message):
    payload = copy.deepcopy(pinned())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(MetalTcError, match=message):
        config_from_mapping(payload)


@pytest.mark.parametrize(
    ("drop", "noun"),
    [
        ({"m": 1, "n": 1, "k": 1}, "1 x 1 x 1"),
        ({"m": 7, "n": 9, "k": 5}, "a K from 2 to 15"),
        ({"m": 257, "n": 131, "k": 1000}, "1000 or more"),
        ({"m": 64, "n": 96, "k": 48}, "multiples of 32"),
    ],
)
def test_the_cases_must_cover_every_edge_of_the_tiling(drop, noun):
    payload = copy.deepcopy(pinned())
    payload["correctness"]["cases"].remove(drop)

    with pytest.raises(MetalTcError, match=noun):
        config_from_mapping(payload)


def test_unknown_fields_duplicate_cases_and_large_cases_are_refused():
    payload = copy.deepcopy(pinned())
    payload["extra"] = 1
    with pytest.raises(MetalTcError, match="schema"):
        config_from_mapping(payload)
    payload = copy.deepcopy(pinned())
    payload["correctness"]["cases"].append({"m": 1, "n": 1, "k": 1})
    with pytest.raises(MetalTcError, match="once"):
        config_from_mapping(payload)
    with pytest.raises(MetalTcError):
        TcCase(2000, 1, 1)
    with pytest.raises(TypeError):
        MetalTcConfig("x", 1, plan_sha256(), None, None, None)
