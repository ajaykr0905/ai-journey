import copy
import json
from pathlib import Path

import pytest

from ai_journey.matmul_checklist_config import (
    SIDES_PER_POINT,
    MatmulChecklistConfig,
    config_from_mapping,
    load_config,
)
from ai_journey.matmul_checklist_source import MatmulChecklistError
from ai_journey.matmul_checklist_steps import checklist_sha256
from ai_journey.matmul_metal_config import load_config as load_day_74_config

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-77-matmul-checklist.json"
# Shapes that run every path: 17 x 33 x 9 stages on the scalar path, and
# 65 x 68 x 40 holds one interior 64 x 64 group, edge groups and a short
# K phase on the vector path.
TEST_CORRECTNESS = [[1, 1, 1], [17, 33, 9], [65, 68, 40]]


def make_config_mapping():
    """A small config whose correctness shapes run every path of every kernel."""

    return {
        "schema_version": 1,
        "name": "matmul-checklist-test",
        "seed": 77,
        "checklist_sha256": checklist_sha256(),
        "naive": {"group_size": [32, 8]},
        "correctness": {"shapes": copy.deepcopy(TEST_CORRECTNESS)},
        "benchmark": {
            "shapes": [[64, 64, 64], [65, 68, 40]],
            "warmup": 1,
            "repeats": 3,
            "target_flops_per_sample": 1 << 20,
            "max_runtime_seconds": 60,
        },
        "comparison": {"min_faster_ratio": 1.1},
    }


def changed(path, value):
    mapping = make_config_mapping()
    *parents, field = path
    target = mapping
    for parent in parents:
        target = target[parent]
    target[field] = value
    return mapping


def test_the_pinned_config_times_the_day_77_shapes_on_the_day_74_cases():
    config = load_config(PINNED)
    day_74 = load_day_74_config(ROOT / "config/day-74-matmul-metal.json")
    benchmark = config.benchmark

    assert config.name == "day-77-matmul-checklist" and config.seed == 77
    assert config.checklist_sha256 == checklist_sha256()
    assert config.correctness.shapes[:-1] == day_74.correctness.shapes
    assert config.correctness.shapes[-1].to_mapping() == {"m": 130, "n": 196, "k": 1000}
    assert config.naive.group_size == day_74.launch.group_size == (32, 8)
    assert [[shape.m, shape.n, shape.k] for shape in benchmark.shapes] == [
        [1024, 1024, 1024],
        [2048, 2048, 2048],
        [4096, 4096, 4096],
        [2000, 3000, 1000],
    ]
    assert (benchmark.warmup, benchmark.repeats) == (2, 11)
    assert benchmark.target_flops_per_sample == 1 << 30
    assert benchmark.max_runtime_seconds == 300
    assert config.comparison.min_faster_ratio == 1.1
    assert config.planned_work().to_mapping() == {
        "correctness_cases": 10,
        "replay_steps": 186189684,
        "correctness_launches": 100,
        "benchmark_points": 4,
        "sides_per_point": SIDES_PER_POINT,
        "timed_samples": 484,
        "calls": 572,
        "flops": 24133581801472,
    }
    assert SIDES_PER_POINT == 11


def test_a_config_round_trips_and_its_fingerprint_binds_every_field():
    mapping = make_config_mapping()
    config = config_from_mapping(mapping)

    assert isinstance(config, MatmulChecklistConfig)
    assert config.to_mapping() == mapping
    assert config_from_mapping(json.loads(json.dumps(mapping))) == config
    assert config_from_mapping(changed(("seed",), 78)).fingerprint() != (
        config.fingerprint()
    )


def test_a_config_for_other_hypotheses_is_refused():
    with pytest.raises(MatmulChecklistError, match="other hypotheses"):
        config_from_mapping(changed(("checklist_sha256",), "0" * 64))
    with pytest.raises(MatmulChecklistError, match="SHA-256 digest"):
        config_from_mapping(changed(("checklist_sha256",), "cb5e"))


@pytest.mark.parametrize(
    ("group", "shapes", "message"),
    [
        ([32, 8], [[17, 32, 9], [65, 64, 24]], "x_columns pads its x axis"),
        ([32, 8], [[17, 40, 9], [65, 72, 24]], "x_rows pads its y axis"),
        ([3, 3], [[64, 65, 9], [128, 68, 24]], "an M that is not a multiple of the "),
        ([32, 8], [[17, 33, 64], [65, 68, 32]], "a K that is not a multiple of the "),
        ([32, 8], [[17, 33, 9], [65, 68, 16]], "a K above the K phase"),
        ([32, 8], [[17, 36, 10], [65, 68, 24], [3, 4, 1000]], "scalar path"),
        ([32, 8], [[17, 33, 9], [64, 64, 24], [3, 5, 1000]], "edge on M or N"),
        ([32, 8], [[17, 33, 9], [63, 68, 24], [3, 5, 1000]], "interior group"),
    ],
)
def test_correctness_shapes_must_run_every_path_of_every_kernel(group, shapes, message):
    mapping = changed(("correctness", "shapes"), shapes)
    mapping["naive"]["group_size"] = group
    with pytest.raises(MatmulChecklistError, match=message):
        config_from_mapping(mapping)


def test_fields_types_and_work_bounds_are_strict():
    mapping = make_config_mapping()
    with pytest.raises(MatmulChecklistError, match="fields do not match"):
        config_from_mapping({**mapping, "variants": []})
    with pytest.raises(MatmulChecklistError, match="fields do not match"):
        config_from_mapping(changed(("benchmark",), {"shapes": []}))
    with pytest.raises(MatmulChecklistError, match="schema_version must be 1"):
        config_from_mapping(changed(("schema_version",), 2))
    with pytest.raises(MatmulChecklistError, match="seed must be an integer"):
        config_from_mapping(changed(("seed",), True))
    with pytest.raises(MatmulChecklistError, match="lowercase slug"):
        config_from_mapping(changed(("name",), "Day 77"))
    with pytest.raises(MatmulChecklistError, match="min_faster_ratio"):
        config_from_mapping(changed(("comparison", "min_faster_ratio"), 1.0))
    with pytest.raises(MatmulChecklistError, match="\\[m, n, k\\] triples"):
        config_from_mapping(changed(("benchmark", "shapes"), [[64, 64]]))
    with pytest.raises(MatmulChecklistError, match="must not repeat a shape"):
        config_from_mapping(changed(("benchmark", "shapes"), [[64, 64, 64]] * 2))
    big = [[8192, 8192, 8192]] * 1 + [[8192, 8192, 8191]]
    with pytest.raises(MatmulChecklistError, match="FLOPs; the bound is"):
        config_from_mapping(changed(("benchmark", "shapes"), big))
    deep = [[1, 1, 1], [17, 33, 9], [65, 68, 40], [8192, 128, 8000]]
    with pytest.raises(MatmulChecklistError, match="replay steps; the bound is"):
        config_from_mapping(changed(("correctness", "shapes"), deep))


def test_a_config_file_is_read_strictly(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(make_config_mapping()), encoding="utf-8")
    assert load_config(path) == config_from_mapping(make_config_mapping())

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"seed": 1, "seed": 2}', encoding="utf-8")
    with pytest.raises(MatmulChecklistError, match="duplicate"):
        load_config(duplicate)
    with pytest.raises(MatmulChecklistError, match="non-symlink"):
        load_config(tmp_path)
