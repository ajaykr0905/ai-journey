import copy
import json
from pathlib import Path

import pytest

from ai_journey.matmul_metal_config import load_config as load_day_74_config
from ai_journey.matmul_tiled_config import (
    MatmulTiledConfig,
    config_from_mapping,
    load_config,
)
from ai_journey.matmul_tiled_source import MatmulTiledError

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-76-matmul-tiled.json"
DAY_76 = ["matmul_tiled_8", "matmul_tiled_16", "matmul_tiled_32"]


def make_config_mapping():
    """A small config whose shapes pad every tile and the naive group."""

    return {
        "schema_version": 1,
        "name": "matmul-tiled-test",
        "seed": 76,
        "naive": {"group_size": [32, 8]},
        "variants": list(DAY_76),
        "correctness": {"shapes": [[1, 1, 1], [17, 33, 9], [31, 47, 129]]},
        "benchmark": {
            "shapes": [[64, 64, 64], [40, 24, 72]],
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


def test_the_pinned_config_times_the_day_76_shapes_on_the_day_74_cases():
    config = load_config(PINNED)
    day_74 = load_day_74_config(ROOT / "config/day-74-matmul-metal.json")
    benchmark = config.benchmark

    assert config.variants == tuple(DAY_76)
    assert config.correctness.shapes == day_74.correctness.shapes
    assert config.naive.group_size == day_74.launch.group_size == (32, 8)
    assert [[shape.m, shape.n, shape.k] for shape in benchmark.shapes] == [
        [1000, 1000, 1000],
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
        "correctness_cases": 9,
        "replay_steps": 160709684,
        "correctness_launches": 36,
        "benchmark_points": 5,
        "sides_per_point": 5,
        "timed_samples": 275,
        "calls": 325,
        "flops": 11099809909760,
    }


def test_a_mapping_round_trips_and_its_fingerprint_binds_every_control():
    config = config_from_mapping(make_config_mapping())
    other = config_from_mapping(changed(("comparison", "min_faster_ratio"), 1.2))

    assert isinstance(config, MatmulTiledConfig)
    assert config.to_mapping() == make_config_mapping()
    assert config_from_mapping(config.to_mapping()) == config
    assert len(config.fingerprint()) == 64
    assert other.fingerprint() != config.fingerprint()
    assert [variant.tile for variant in config.tiled_variants()] == [8, 16, 32]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), 2, "schema_version"),
        (("name",), "Matmul Tiled", "slug"),
        (("seed",), True, "seed must be an integer"),
        (("naive",), {"group_size": [32, 8], "extra": 1}, "naive fields"),
        (("naive", "group_size"), [32, 8, 1], "group_size"),
        (("variants",), "matmul_tiled_8", "JSON array"),
        (("variants",), ["matmul_tiled_8", "matmul_tiled_8"], "twice"),
        (("variants",), ["matmul_tiled_12"], "not a registered variant"),
        (("variants",), [], "1 to 8 variants"),
        (("correctness", "shapes"), [[17, 33]], "triples"),
        (("correctness", "shapes"), [[17, 33, 9], [17, 33, 9]], "repeat"),
        (("benchmark", "warmup"), 11, "warmup"),
        (("benchmark", "repeats"), 2, "repeats"),
        (("benchmark", "target_flops_per_sample"), 1.5, "integer"),
        (("benchmark", "max_runtime_seconds"), 601, "max_runtime_seconds"),
        (("comparison", "min_faster_ratio"), 1, "finite float"),
        (("comparison", "min_faster_ratio"), 1.0, "above 1"),
        (("comparison", "min_faster_ratio"), 4.5, "at most 4.0"),
    ],
)
def test_malformed_controls_are_refused_with_one_error_type(path, value, message):
    with pytest.raises(MatmulTiledError, match=message):
        config_from_mapping(changed(path, value))


def test_missing_and_unknown_fields_are_refused():
    mapping = make_config_mapping()
    del mapping["comparison"]
    with pytest.raises(MatmulTiledError, match="config fields"):
        config_from_mapping(mapping)
    mapping = make_config_mapping()
    mapping["benchmark"]["device"] = "mps"
    with pytest.raises(MatmulTiledError, match="benchmark fields"):
        config_from_mapping(mapping)


@pytest.mark.parametrize(
    ("shapes", "message"),
    [
        ([[16, 33, 9], [32, 47, 129]], "an M that is not a multiple of tile 8"),
        ([[17, 32, 9], [31, 48, 129]], "an N that is not a multiple of tile 8"),
        ([[17, 33, 8], [31, 47, 128]], "a K that is not a multiple of tile 8"),
        ([[17, 33, 9], [31, 47, 31]], "a K above tile 32"),
        ([[17, 33, 9], [31, 47, 15]], "a K above tile 16"),
    ],
)
def test_every_tile_must_pad_both_axes_and_end_a_later_phase_early(shapes, message):
    with pytest.raises(MatmulTiledError, match=message):
        config_from_mapping(changed(("correctness", "shapes"), shapes))


def test_one_tile_needs_only_its_own_padding():
    shapes = [[9, 9, 9], [24, 8, 16]]
    mapping = changed(("correctness", "shapes"), shapes)
    mapping["variants"] = ["matmul_tiled_8"]
    mapping["naive"]["group_size"] = [8, 8]

    config = config_from_mapping(mapping)

    assert config.planned_work().correctness_launches == 4
    assert config.planned_work().sides_per_point == 3


def test_the_naive_group_must_pad_both_axes_too():
    mapping = changed(("correctness", "shapes"), [[17, 33, 9], [31, 47, 129]])
    mapping["naive"]["group_size"] = [1, 8]
    with pytest.raises(MatmulTiledError, match="naive group x size 1"):
        config_from_mapping(mapping)


def test_work_beyond_the_day_74_bounds_is_refused_before_anything_runs():
    huge = changed(("benchmark", "shapes"), [[8192, 8192, 8192]])
    huge["benchmark"]["target_flops_per_sample"] = 1 << 40
    huge["benchmark"]["repeats"] = 50
    with pytest.raises(MatmulTiledError, match="FLOPs; the bound"):
        config_from_mapping(huge)
    long = changed(
        ("correctness", "shapes"),
        [[17, 33, 9], [31, 47, 129], [4095, 4095, 4095]],
    )
    with pytest.raises(MatmulTiledError, match="replay steps; the bound"):
        config_from_mapping(long)


def test_a_config_file_is_read_strictly(tmp_path):
    good = tmp_path / "config.json"
    good.write_text(json.dumps(make_config_mapping()), encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"seed": 1, "seed": 2}', encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(good)

    assert load_config(good) == config_from_mapping(
        copy.deepcopy(make_config_mapping())
    )
    with pytest.raises(MatmulTiledError, match="duplicate JSON field"):
        load_config(duplicate)
    with pytest.raises(MatmulTiledError, match="non-symlink"):
        load_config(link)
