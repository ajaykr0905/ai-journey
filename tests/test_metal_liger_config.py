import copy
import json
from pathlib import Path

import pytest

from ai_journey.metal_liger_config import (
    COMPULSORY,
    compulsory_bytes,
    config_from_mapping,
    load_config,
)
from ai_journey.metal_liger_plan import plan_sha256
from ai_journey.metal_liger_reference import dense_layout
from ai_journey.metal_liger_source import (
    CROSS_ENTROPY,
    FAMILIES,
    RMSNORM,
    SWIGLU,
    MetalLigerError,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "day-80-metal-liger.json"


def payload():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_the_day_80_config_pins_the_plan_and_the_small_llm_shapes():
    config = load_config(CONFIG)

    assert config.plan_sha256 == plan_sha256()
    shapes = config.benchmark.shapes
    assert [shape.to_mapping() for shape in shapes[RMSNORM]] == [
        {"rows": 4096, "cols": 1024},
        {"rows": 4096, "cols": 4096},
        {"rows": 4096, "cols": 2049},
    ]
    assert shapes[SWIGLU] == tuple(
        type(shape)(SWIGLU, shape.rows, shape.cols) for shape in shapes[RMSNORM]
    )
    assert [(shape.rows, shape.cols) for shape in shapes[CROSS_ENTROPY]] == [
        (1024, 32000),
        (1024, 50257),
    ]
    assert config.comparison.min_faster_ratio == 1.1
    assert config.benchmark.repeats % 2 == 0
    assert config.required_threads() == 1024
    work = config.planned_work()
    assert work["benchmark_points"] == 8 and work["timed_samples"] == 480
    assert config_from_mapping(config.to_mapping()) == config
    assert config.fingerprint() == config_from_mapping(payload()).fingerprint()


def test_compulsory_bytes_follow_the_declared_rules():
    layout = dense_layout(4, 100)

    assert compulsory_bytes(RMSNORM, layout) == 8 * 400 + 4 * 100
    assert compulsory_bytes(SWIGLU, layout) == 12 * 400
    assert compulsory_bytes(CROSS_ENTROPY, layout) == 4 * 400 + 8 * 4
    assert set(COMPULSORY) == set(FAMILIES)
    with pytest.raises(MetalLigerError, match="kernel family"):
        compulsory_bytes("layernorm", layout)


def _edit(change):
    data = payload()
    change(data)
    return data


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(plan_sha256="0" * 64), "declared plan"),
        (lambda d: d.update(name="Day 80"), "lowercase slug"),
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d.update(extra=1), "fields do not match"),
        (lambda d: d["benchmark"].update(repeats=31), "multiple of the 2"),
        (lambda d: d["benchmark"].update(inputs="cauchy"), "inputs must be one of"),
        (lambda d: d["benchmark"]["shapes"].pop(SWIGLU), "fields do not match"),
        (lambda d: d["benchmark"]["shapes"].update(swiglu=[]), "1 to 4 shapes"),
        (
            lambda d: d["benchmark"]["shapes"][RMSNORM].append(
                {"rows": 2, "cols": 16385}
            ),
            "longest row",
        ),
        (
            lambda d: d["benchmark"]["shapes"][RMSNORM].__setitem__(
                0, {"rows": 2, "max_length": 9}
            ),
            "benchmark shape fields",
        ),
        (lambda d: d["correctness"].update(repeats=1), "repeats"),
        (
            lambda d: d["correctness"]["cases"][CROSS_ENTROPY].pop(0),
            "need a row of 1 element",
        ),
        (
            lambda d: d["correctness"]["cases"].update(
                cross_entropy=[
                    {"lengths": [1, 31, 33, 1000, 7], "inputs": "uniform_80"}
                ]
            ),
            "a row longer than 16384",
        ),
        (
            lambda d: d["correctness"]["cases"][SWIGLU].__setitem__(
                0, {"lengths": [5, 5], "inputs": "normal"}
            ),
            "two or more lengths",
        ),
        (
            lambda d: [
                case.update(inputs="normal")
                for case in d["correctness"]["cases"][RMSNORM]
            ],
            "large-magnitude inputs",
        ),
        (lambda d: d["comparison"].update(min_faster_ratio=0.9), "min_faster_ratio"),
        (lambda d: d["benchmark"].update(warmup=True), "warmup"),
    ],
)
def test_the_config_refuses_a_change_that_breaks_its_contract(change, message):
    with pytest.raises((MetalLigerError, TypeError), match=message):
        config_from_mapping(_edit(change))


def test_a_config_file_with_a_duplicate_key_is_refused(tmp_path):
    path = tmp_path / "config.json"
    text = CONFIG.read_text(encoding="utf-8").replace(
        '"seed": 80,', '"seed": 80, "seed": 81,'
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(MetalLigerError, match="duplicate"):
        load_config(path)
    data = copy.deepcopy(payload())
    assert load_config(CONFIG).to_mapping() == data
