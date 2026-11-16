import json
from pathlib import Path

import pytest

from ai_journey.kernel_metal_benchmark import BENCHMARK_INDEX_OFFSET, MAX_POINTS
from ai_journey.kernel_metal_config import (
    MAX_CONFIG_BYTES,
    PlannedWork,
    config_from_mapping,
    load_config,
)
from ai_journey.kernel_metal_geometry import KernelMetalError

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-73-kernel-metal.json"


def make_config_mapping() -> dict:
    return {
        "schema_version": 1,
        "name": "kernel-metal-test",
        "seed": 73,
        "launch": {"vector_add_group_size": 256, "rgb_to_gray_group_size": [16, 16]},
        "correctness": {
            "vector_add_lengths": [1, 257, 4096],
            "rgb_to_gray_shapes": [[1, 1], [17, 33]],
            "rgb_to_gray_all_triples": False,
        },
        "benchmark": {
            "vector_add_lengths": [4096, 65537],
            "rgb_to_gray_shapes": [[17, 33], [64, 64]],
            "warmup": 1,
            "repeats": 3,
            "target_bytes_per_sample": 65536,
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
    assert config.launch.rgb_to_gray_group_size == (16, 16)
    assert config.correctness.rgb_to_gray_shapes == ((1, 1), (17, 33))
    assert (
        config.fingerprint() == config_from_mapping(make_config_mapping()).fingerprint()
    )
    assert (
        config.fingerprint()
        != config_from_mapping(changed(("seed",), 74)).fingerprint()
    )
    assert config.correctness.gray_cases() == (
        (0, (1, 1), "uniform"),
        (1, (17, 33), "uniform"),
    )
    with_triples = config_from_mapping(
        changed(("correctness", "rgb_to_gray_all_triples"), True)
    )
    assert with_triples.correctness.gray_cases()[-1] == (
        2,
        (4096, 4096),
        "all_rgb_triples",
    )


def test_planned_work_counts_both_sides_of_every_sample():
    work = config_from_mapping(make_config_mapping()).planned_work()

    # Calls per sample: 2 and 1 for the vectors, 30 and 4 for the images.
    assert work == PlannedWork(
        correctness_cases=5,
        benchmark_points=4,
        timed_samples=24,
        calls=2 * 4 * (2 + 1 + 30 + 4),
        accounted_bytes=2
        * 4
        * (2 * 12 * 4096 + 12 * 65537 + 30 * 4 * 17 * 33 + 4 * 4 * 64 * 64),
    )
    assert work.to_mapping()["timed_samples"] == 24


@pytest.mark.parametrize(
    ("path", "value", "error", "message"),
    [
        (("schema_version",), 2, KernelMetalError, "schema_version must be 1"),
        (("extra",), 1, KernelMetalError, "config fields"),
        (("seed",), KeyError, KernelMetalError, "config fields"),
        (("launch", "extra"), 1, KernelMetalError, "launch fields"),
        (
            ("correctness", "vector_add_lengths"),
            KeyError,
            KernelMetalError,
            "correctness",
        ),
        (("benchmark", "warmup"), KeyError, KernelMetalError, "benchmark fields"),
        (("name",), "Kernel Metal", KernelMetalError, "lowercase slug"),
        (("seed",), True, KernelMetalError, "seed must be an integer"),
        (("seed",), 2**32, KernelMetalError, "seed must be in"),
        (("launch", "vector_add_group_size"), 2048, KernelMetalError, "group_size"),
        (("launch", "vector_add_group_size"), 256.0, KernelMetalError, "integer"),
        (("launch", "rgb_to_gray_group_size"), [64, 32], KernelMetalError, "1024"),
        (("launch", "rgb_to_gray_group_size"), [16], KernelMetalError, "pair"),
        (("launch", "rgb_to_gray_group_size"), 16, KernelMetalError, "JSON array"),
        # A size of 1 divides every extent, so no case could test the guard.
        (("launch", "vector_add_group_size"), 1, KernelMetalError, "group size 1"),
        (("launch", "rgb_to_gray_group_size"), [1, 16], KernelMetalError, "group x"),
        (("launch", "rgb_to_gray_group_size"), [16, 1], KernelMetalError, "group y"),
        (
            ("correctness", "vector_add_lengths"),
            [257, 1],
            KernelMetalError,
            "strictly increasing",
        ),
        (("correctness", "vector_add_lengths"), [], KernelMetalError, "1 to 16"),
        (
            ("correctness", "vector_add_lengths"),
            [256, 512],
            KernelMetalError,
            "not a multiple of the group size 256",
        ),
        (
            ("correctness", "rgb_to_gray_shapes"),
            [[16, 33], [32, 17]],
            KernelMetalError,
            "height that is not a multiple",
        ),
        (
            ("correctness", "rgb_to_gray_shapes"),
            [[17, 33], [17, 33]],
            KernelMetalError,
            "repeat a shape",
        ),
        (
            ("correctness", "rgb_to_gray_shapes"),
            [[17, 33, 3]],
            KernelMetalError,
            r"\[height, width\] pairs",
        ),
        (
            ("correctness", "rgb_to_gray_shapes"),
            [[0, 33]],
            KernelMetalError,
            "height must be in",
        ),
        (
            ("correctness", "rgb_to_gray_all_triples"),
            1,
            KernelMetalError,
            "true or false",
        ),
        (
            ("benchmark", "rgb_to_gray_shapes"),
            [[64, 64], [32, 128]],
            KernelMetalError,
            "grow in pixels",
        ),
        (("benchmark", "repeats"), 2, KernelMetalError, "repeats must be in"),
        (("benchmark", "warmup"), 11, KernelMetalError, "warmup must be in"),
        (
            ("benchmark", "target_bytes_per_sample"),
            1 << 31,
            KernelMetalError,
            "target_bytes_per_sample",
        ),
        (
            ("benchmark", "target_bytes_per_sample"),
            1 << 26,
            KernelMetalError,
            "at most 4096 are allowed",
        ),
        (("benchmark", "max_runtime_seconds"), 0, KernelMetalError, "max_runtime"),
    ],
)
def test_invalid_controls_are_rejected(path, value, error, message):
    with pytest.raises(error, match=message):
        config_from_mapping(changed(path, value))


def test_config_files_are_bounded_strict_and_duplicate_free(tmp_path):
    path = tmp_path / "config.json"

    for payload, message in (
        (b'{"seed": 1, "seed": 2}', "duplicate JSON field: seed"),
        (b'{"seed": NaN}', "non-finite"),
        (b"[]", "root must be a JSON object"),
        (b" " * (MAX_CONFIG_BYTES + 1), "exceeds"),
    ):
        path.write_bytes(payload)
        with pytest.raises(KernelMetalError, match=message):
            load_config(path)
    path.write_text(json.dumps(make_config_mapping()), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(KernelMetalError, match="non-symlink"):
        load_config(link)
    assert load_config(path).to_mapping() == make_config_mapping()


def test_the_pinned_config_parses_to_the_checked_in_mapping():
    config = load_config(PINNED)

    assert config.to_mapping() == json.loads(PINNED.read_text(encoding="utf-8"))
    assert (config.name, config.seed) == ("day-73-kernel-metal", 73)
    assert config.launch.vector_add_group_size == 256
    assert config.launch.rgb_to_gray_group_size == (16, 16)
    assert config.correctness.rgb_to_gray_all_triples is True
    assert config.planned_work().to_mapping() == {
        "correctness_cases": 13,
        "benchmark_points": 13,
        "timed_samples": 286,
        "calls": 81354,
        "accounted_bytes": 17363214336,
    }


def test_the_pinned_cases_exercise_the_guards_and_the_sweep_spans_sizes():
    config = load_config(PINNED)
    lengths = config.correctness.vector_add_lengths
    shapes = config.correctness.rgb_to_gray_shapes

    assert {length % 256 for length in lengths} - {0}
    assert 256 in lengths
    assert any(width % 16 and height % 16 for height, width in shapes)
    assert (16, 16) in shapes
    vectors = config.benchmark.vector_add_lengths
    assert (12 * vectors[0], 12 * vectors[-1]) == (48 * 1024, 192 * 1024 * 1024)
    images = config.benchmark.rgb_to_gray_shapes
    assert (1080, 1920) in images
    assert images[-1] == (4096, 4096)
    # Correctness cases, all triples included, never reach the benchmark streams.
    assert MAX_POINTS + 1 <= BENCHMARK_INDEX_OFFSET
