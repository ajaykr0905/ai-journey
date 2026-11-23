import copy
import json
from pathlib import Path

import pytest

from ai_journey.gpu_reduce_config import (
    SIDES_PER_POINT,
    GpuReduceConfig,
    PlannedWork,
    config_from_mapping,
    load_config,
)
from ai_journey.gpu_reduce_plan import plan_sha256
from ai_journey.gpu_reduce_source import GpuReduceError

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-78-gpu-reduce.json"
# Small cases that still run every path, each rule from one case: n = 1 fits
# one SIMD group, 300007 ends its last 512-element group in the second half,
# and 1600001 gives every kernel but R5 more than 3072 partials.
TEST_CASES = [
    ["normal", 1],
    ["normal", 33],
    ["positive", 300007],
    ["mixed", 1600001],
    ["ones", 1025],
    ["halves", 2049],
]


def make_config_mapping():
    """A small config whose cases run every path of every kernel."""

    return {
        "schema_version": 1,
        "name": "gpu-reduce-test",
        "seed": 78,
        "plan_sha256": plan_sha256(),
        "correctness": {"cases": copy.deepcopy(TEST_CASES), "runs": 2},
        "benchmark": {
            "sizes": [4096, 70001],
            "warmup": 1,
            "repeats": 3,
            "target_bytes_per_sample": 1 << 20,
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


def test_the_pinned_config_times_the_day_78_sizes_on_every_path():
    config = load_config(PINNED)

    assert config.name == "day-78-gpu-reduce" and config.seed == 78
    assert config.plan_sha256 == plan_sha256()
    assert config.benchmark.sizes == (1 << 16, 1 << 20, 1 << 24, 1 << 26, 10000019)
    assert (config.benchmark.warmup, config.benchmark.repeats) == (2, 14)
    # Repeats are a multiple of the 7 sides, so each side takes each place
    # of the rotation equally often.
    assert config.benchmark.repeats % SIDES_PER_POINT == 0
    assert config.benchmark.target_bytes_per_sample == 1 << 30
    assert [config.benchmark.calls_per_sample(n) for n in config.benchmark.sizes] == [
        4096,
        256,
        16,
        4,
        27,
    ]
    assert config.comparison.min_faster_ratio == 1.1
    assert [case.n for case in config.correctness.cases][:4] == [1, 31, 33, 257]
    assert {case.pattern for case in config.correctness.cases} == {
        "normal",
        "positive",
        "mixed",
        "ones",
        "halves",
    }
    assert config.planned_work() == PlannedWork(
        correctness_cases=13,
        correctness_elements=7301371,
        correctness_launches=468,
        benchmark_points=5,
        sides_per_point=7,
        timed_samples=490,
        calls=492688,
        bytes_read=601996566976,
    )
    assert config.to_mapping() == json.loads(PINNED.read_text(encoding="utf-8"))


def test_the_test_config_is_valid_and_has_a_stable_fingerprint():
    config = config_from_mapping(make_config_mapping())

    assert isinstance(config, GpuReduceConfig)
    assert config.to_mapping() == make_config_mapping()
    assert (
        config.fingerprint() == config_from_mapping(make_config_mapping()).fingerprint()
    )
    assert (
        config.fingerprint()
        != config_from_mapping(changed(("seed",), 79)).fingerprint()
    )
    assert config.planned_work().correctness_launches == 6 * 6 * 2 * 2


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (["ones", 1025], None, "the pattern ones"),
        (["normal", 1], ["normal", 64], "an n below 32"),
        (["positive", 300007], ["positive", 600], "ends in its second half"),
        (["mixed", 1600001], ["mixed", 1000003], "more than 3072 partials"),
    ],
)
def test_cases_that_skip_a_pattern_or_a_path_are_refused(old, new, message):
    cases = [case for case in TEST_CASES if case != old]
    if new is not None:
        cases.append(new)

    with pytest.raises(GpuReduceError, match=message):
        config_from_mapping(changed(("correctness", "cases"), cases))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("plan_sha256",), "0" * 64, "does not name the declared plan"),
        (("plan_sha256",), "abc", "SHA-256 digest"),
        (("name",), "Day 78", "lowercase slug"),
        (("seed",), -1, r"seed must be in"),
        (("seed",), 1.5, "seed must be an integer"),
        (("schema_version",), 2, "schema_version must be 1"),
        (("correctness", "runs"), 1, r"runs must be in \[2, 5\]"),
        (("correctness", "cases"), [["normal", 1, 2]], r"\[pattern, n\] pair"),
        (("correctness", "cases"), [["uniform", 9]], "pattern must be one of"),
        (
            ("correctness", "cases"),
            [*TEST_CASES, ["normal", 1]],
            "only once",
        ),
        (
            ("correctness", "cases"),
            [*TEST_CASES, ["normal", (1 << 22) + 1]],
            r"case n must be in",
        ),
        (("benchmark", "sizes"), [4096, 4096], "only once"),
        (("benchmark", "sizes"), [(1 << 26) + 1], r"benchmark n must be in"),
        (("benchmark", "sizes"), [63], "at most 4096 are allowed"),
        (("benchmark", "repeats"), 2, r"repeats must be in \[3, 50\]"),
        (("benchmark", "warmup"), 11, r"warmup must be in \[0, 10\]"),
        (("benchmark", "max_runtime_seconds"), 601, "max_runtime_seconds"),
        (("comparison", "min_faster_ratio"), 1.0, "min_faster_ratio must be above"),
        (("comparison", "min_faster_ratio"), 1, "finite float"),
    ],
)
def test_a_wrong_control_is_refused_as_this_labs_error(path, value, message):
    with pytest.raises(GpuReduceError, match=message):
        config_from_mapping(changed(path, value))


def test_unknown_fields_extra_work_and_duplicate_keys_are_refused(tmp_path):
    mapping = make_config_mapping()
    mapping["benchmark"]["shapes"] = []
    with pytest.raises(GpuReduceError, match="benchmark fields"):
        config_from_mapping(mapping)
    big = changed(
        ("benchmark",),
        {
            "sizes": [(1 << 26) - index for index in range(16)],
            "warmup": 10,
            "repeats": 50,
            "target_bytes_per_sample": 1 << 30,
            "max_runtime_seconds": 600,
        },
    )
    with pytest.raises(GpuReduceError, match="bytes; the bound is"):
        config_from_mapping(big)
    path = tmp_path / "config.json"
    path.write_text('{"seed": 1, "seed": 2}', encoding="utf-8")
    with pytest.raises(GpuReduceError, match="duplicate JSON field"):
        load_config(path)
    with pytest.raises(GpuReduceError, match="non-symlink file"):
        load_config(tmp_path)
