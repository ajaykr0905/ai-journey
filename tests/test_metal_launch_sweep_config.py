import json
from pathlib import Path

import pytest

from ai_journey.metal_launch_sweep_config import (
    MAX_CONFIG_BYTES,
    PlannedWork,
    config_from_mapping,
    load_config,
)
from ai_journey.metal_launch_sweep_shapes import MetalLaunchSweepError

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-74-metal-launch-sweep.json"


def make_config_mapping() -> dict:
    return {
        "schema_version": 1,
        "name": "metal-launch-sweep-test",
        "seed": 74,
        "matmul": {
            "group_shapes": [[8, 8], [16, 16], [4, 64]],
            "correctness_cases": [[1, 1, 1], [37, 53, 29]],
            "sizes": [[32, 32, 32], [64, 64, 64]],
            "target_flops_per_sample": 1 << 20,
        },
        "vector": {
            "group_sizes": [8, 32, 48, 256],
            "correctness_length": 1001,
            "lengths": [4096, 65537],
            "target_bytes_per_sample": 1 << 16,
        },
        "timing": {"warmup": 1, "repeats": 3, "max_runtime_seconds": 60},
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
    config = config_from_mapping(make_config_mapping())

    assert config.to_mapping() == make_config_mapping()
    assert config.matmul.group_shapes == ((8, 8), (16, 16), (4, 64))
    assert config.matmul.sizes == ((32, 32, 32), (64, 64, 64))
    assert config.vector.group_sizes == (8, 32, 48, 256)
    assert config.matmul_batches() == (16, 2)
    assert config.vector_batches() == (2, 1)
    assert (
        config.fingerprint() == config_from_mapping(make_config_mapping()).fingerprint()
    )
    assert (
        config.fingerprint()
        != config_from_mapping(changed(("seed",), 75)).fingerprint()
    )


def test_planned_work_counts_every_shape_as_admitted():
    work = config_from_mapping(make_config_mapping()).planned_work()

    samples = 1 + 3
    matmul_calls = 3 * samples * (16 + 2)
    vector_calls = 4 * samples * (2 + 1)
    assert work == PlannedWork(
        mapping_checks=3,
        layout_probes=3 + 4,
        correctness_launches=3 * 2 + 4,
        benchmark_points=4,
        timed_samples=3 * (3 * 2 + 4 * 2),
        calls=matmul_calls + vector_calls,
        matmul_flops=3 * samples * (16 * 2 * 32**3 + 2 * 2 * 64**3),
        vector_bytes=4 * samples * (2 * 12 * 4096 + 12 * 65537),
    )


def test_the_pinned_config_sweeps_the_seven_shapes_and_nine_sizes():
    config = load_config(PINNED)

    assert config.matmul.group_shapes == (
        (8, 8),
        (16, 16),
        (32, 32),
        (32, 8),
        (8, 32),
        (64, 4),
        (4, 64),
    )
    assert config.vector.group_sizes == (8, 16, 32, 48, 64, 128, 256, 512, 1024)
    assert [size[0] for size in config.matmul.sizes] == [256, 1024, 2048]
    assert config.planned_work().to_mapping() == {
        "mapping_checks": 7,
        "layout_probes": 16,
        "correctness_launches": 30,
        "benchmark_points": 5,
        "timed_samples": 819,
        "calls": 7958,
        "matmul_flops": 3284576239616,
        "vector_bytes": 70325895168,
    }
    # Each shape takes each turn position three times.
    assert config.timing.repeats % len(config.matmul.group_shapes) == 0


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("extra",), 1, "config fields"),
        (("seed",), True, "seed must be an integer"),
        (("seed",), -1, "seed must be in"),
        (("name",), "Bad Name", "lowercase slug"),
        (("schema_version",), 2, "schema_version must be 1"),
        (("matmul", "sizes"), KeyError, "matmul fields"),
        (("matmul", "group_shapes"), [[8, 8], [8, 8]], "must not repeat"),
        (("matmul", "group_shapes"), [[1, 64], [8, 8]], "2 or more threads"),
        (("matmul", "group_shapes"), [[64, 32]], "2048 threads"),
        (("matmul", "group_shapes"), [[8, 8, 1]], "tuple of 2 sizes"),
        (("matmul", "group_shapes"), [], "1 to 16 items"),
        (("matmul", "correctness_cases"), [[64, 64, 8]], "not a multiple of 8"),
        (("matmul", "correctness_cases"), [[513, 1, 1]], "m must be in"),
        (("matmul", "correctness_cases"), [[3, 3]], "[m, n, k] triples"),
        (("matmul", "sizes"), [[64, 64, 64], [32, 32, 32]], "grow in FLOPs"),
        (("matmul", "sizes"), [[4097, 8, 8]], "m must be in"),
        (("matmul", "sizes"), [[8, 8, n] for n in (1, 2, 3, 4, 5)], "at most 4"),
        (("matmul", "target_flops_per_sample"), 1 << 41, "target_flops"),
        (("matmul", "target_flops_per_sample"), 1.0e6, "must be an integer"),
        (("vector", "group_sizes"), [32, 8], "increasing"),
        (("vector", "group_sizes"), [1, 32], "size of 1 divides"),
        (("vector", "group_sizes"), [2048], "must be in"),
        (("vector", "correctness_length"), 1024, "not be a multiple"),
        (("vector", "lengths"), [65537, 4096], "increasing"),
        (("vector", "lengths"), [1 << 26], "must be in"),
        (("vector", "target_bytes_per_sample"), 512, "target_bytes_per_sample"),
        (("timing", "repeats"), 2, "repeats must be in"),
        (("timing", "warmup"), 11, "warmup must be in"),
        (("timing", "max_runtime_seconds"), 601, "max_runtime_seconds"),
    ],
)
def test_config_refuses_each_broken_control(path, value, message):
    with pytest.raises(MetalLaunchSweepError, match=message.replace("[", "\\[")):
        config_from_mapping(changed(path, value))


def test_config_refuses_oversized_batches_and_runs():
    tiny = changed(("matmul", "sizes"), [[2, 2, 2]])
    with pytest.raises(MetalLaunchSweepError, match="at most 4096 are allowed"):
        config_from_mapping(tiny)
    heavy = changed(("matmul", "sizes"), [[4096, 4096, 4096]])
    heavy["matmul"]["target_flops_per_sample"] = 1 << 40
    heavy["timing"]["repeats"] = 50
    with pytest.raises(MetalLaunchSweepError, match="matmul FLOPs; the bound is"):
        config_from_mapping(heavy)


def test_config_files_are_bounded_strict_regular_json(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(make_config_mapping()), encoding="utf-8")
    assert load_config(good).name == "metal-launch-sweep-test"
    link = tmp_path / "link.json"
    link.symlink_to(good)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"seed": 1, "seed": 2}', encoding="utf-8")
    large = tmp_path / "large.json"
    large.write_text(" " * (MAX_CONFIG_BYTES + 1), encoding="utf-8")
    for path, message in (
        (link, "regular non-symlink"),
        (duplicate, "duplicate JSON field"),
        (large, f"exceeds {MAX_CONFIG_BYTES} bytes"),
        (tmp_path / "missing.json", "regular non-symlink"),
    ):
        with pytest.raises(MetalLaunchSweepError, match=message):
            load_config(path)
