import json
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.matmul_metal_config import (
    MAX_CONFIG_BYTES,
    PlannedWork,
    config_from_mapping,
    load_config,
)
from ai_journey.matmul_metal_mapping import MatmulMetalError, matmul_shape
from ai_journey.memory_access_plan import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-74-matmul-metal.json"


def make_config_mapping() -> dict:
    return {
        "schema_version": 1,
        "name": "matmul-metal-test",
        "seed": 74,
        "launch": {"group_size": [32, 8]},
        "correctness": {"shapes": [[1, 1, 1], [17, 33, 9], [32, 32, 16]]},
        "benchmark": {
            "shapes": [[33, 17, 9], [64, 64, 64]],
            "warmup": 1,
            "repeats": 3,
            "target_flops_per_sample": 1048576,
            "max_runtime_seconds": 60,
        },
    }


def changed(path, value):
    mapping = make_config_mapping()
    target = mapping
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return mapping


def test_config_round_trips_to_its_canonical_mapping_and_fingerprint():
    mapping = make_config_mapping()
    config = config_from_mapping(mapping)

    assert config.to_mapping() == mapping
    assert config.launch.group_size == (32, 8)
    assert config.correctness.shapes[1] == matmul_shape(17, 33, 9)
    assert config.fingerprint() == sha256(canonical_json_bytes(mapping)).hexdigest()
    # 104 calls of 2 * 33 * 17 * 9 FLOPs, or 2 calls of 2 * 64**3, reach 2**20.
    assert config.planned_work() == PlannedWork(
        correctness_cases=3,
        replay_steps=1 + 17 * 33 * 9 + 32 * 32 * 16,
        benchmark_points=2,
        timed_samples=3 * 2 * 3,
        calls=3 * 4 * (104 + 2),
        flops=3 * 4 * (104 * 10098 + 2 * 524288),
    )


def test_the_pinned_config_sweeps_squares_and_non_square_shapes():
    config = load_config(PINNED)
    work = config.planned_work()

    assert config.name == "day-74-matmul-metal"
    squares = [shape.m for shape in config.benchmark.shapes if shape.m == shape.n]
    assert squares == [128, 256, 512, 1024, 2048, 4096]
    assert len(config.benchmark.shapes) - len(squares) == 3
    assert (work.correctness_cases, work.benchmark_points) == (9, 9)
    assert work.timed_samples == 3 * 9 * config.benchmark.repeats
    assert max(shape.k for shape in config.correctness.shapes) == 4096


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("unknown",), 1, "config fields"),
        (("launch", "group_size"), KeyError, "launch fields"),
        (("seed",), True, "seed must be an integer"),
        (("seed",), -1, "seed must be in"),
        (("schema_version",), 2, "schema_version must be 1"),
        (("name",), "Matmul", "lowercase slug"),
        (("launch", "group_size"), [32], r"an \[x, y\] pair"),
        (("launch", "group_size"), [0, 8], "group_size x must be in"),
        (("launch", "group_size"), [64, 32], "more than 1024 threads"),
        (("correctness", "shapes"), [], "1 to 16 shapes"),
        (("correctness", "shapes"), [[17, 33]], r"\[m, n, k\] triples"),
        (("correctness", "shapes"), [[17, 33, 0]], "k must be in"),
        (("correctness", "shapes"), [[17, 33, 9.0]], "k must be an integer"),
        (("correctness", "shapes"), [[17, 33, 9], [17, 33, 9]], "repeat a shape"),
        (("benchmark", "shapes"), "64", "must be a JSON array"),
        (("benchmark", "warmup"), 11, "warmup must be in"),
        (("benchmark", "repeats"), 2, "repeats must be in"),
        (("benchmark", "repeats"), 3.0, "repeats must be an integer"),
        (("benchmark", "target_flops_per_sample"), 1, "target_flops_per_sample"),
        (("benchmark", "max_runtime_seconds"), 601, "max_runtime_seconds must be"),
    ],
)
def test_fields_types_and_ranges_are_strict(path, value, message):
    with pytest.raises(KernelMetalError, match=message):
        config_from_mapping(changed(path, value))


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        (
            "correctness",
            {"shapes": [[32, 33, 1], [64, 17, 2]]},
            "an M that is not a multiple of the group x size 32",
        ),
        (
            "correctness",
            {"shapes": [[17, 32, 1], [9, 64, 2]]},
            "an N that is not a multiple of the group x size 32",
        ),
        ("correctness", {"shapes": [[17, 17, 3], [9, 9, 2]]}, "non-square"),
        (
            "correctness",
            {"shapes": [[17, 33, 9], [8191, 4097, 1]]},
            "with its guard tail; the bound is 33555455",
        ),
        (
            "correctness",
            {"shapes": [[17, 33, 9], [1023, 1025, 512]]},
            "replay steps; the bound is 268435456",
        ),
    ],
)
def test_the_whole_config_must_exercise_both_guards_within_bounds(
    section, value, message
):
    with pytest.raises(MatmulMetalError, match=message):
        config_from_mapping({**make_config_mapping(), section: value})


def test_sweeps_must_fit_the_batch_and_work_bounds():
    benchmark = make_config_mapping()["benchmark"]
    tiny = {**benchmark, "shapes": [[1, 1, 1]], "target_flops_per_sample": 1 << 40}
    huge = {**benchmark, "shapes": [[8192, 8192, 8192]], "repeats": 50}

    with pytest.raises(MatmulMetalError, match="at most 4096 are allowed"):
        config_from_mapping({**make_config_mapping(), "benchmark": tiny})
    with pytest.raises(MatmulMetalError, match="FLOPs; the bound is"):
        config_from_mapping({**make_config_mapping(), "benchmark": huge})


def test_config_files_are_read_strictly(tmp_path):
    good = tmp_path / "config.json"
    good.write_text(json.dumps(make_config_mapping()), encoding="utf-8")
    assert load_config(good).name == "matmul-metal-test"

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"seed": 1, "seed": 2}', encoding="utf-8")
    large = tmp_path / "large.json"
    large.write_text(" " * (MAX_CONFIG_BYTES + 1), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(good)
    listed = tmp_path / "list.json"
    listed.write_text("[]", encoding="utf-8")
    for path, message in (
        (duplicate, "duplicate JSON field"),
        (large, "exceeds"),
        (link, "regular non-symlink"),
        (listed, "root must be a JSON object"),
        (tmp_path / "missing.json", "regular non-symlink"),
    ):
        with pytest.raises(MatmulMetalError, match=message):
            load_config(path)
