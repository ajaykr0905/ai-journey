import copy
import json

import pytest

from ai_journey.gpu_scan_config import (
    MAX_CASE_ELEMENTS,
    PlannedWork,
    config_from_mapping,
    load_config,
    scan_calls_per_sample,
)
from ai_journey.gpu_scan_plan import plan_sha256
from ai_journey.gpu_scan_source import GpuScanError

NORMAL, INTEGERS = "standard_normal", "small_integers"


def make_config_mapping():
    """A short config that still runs a ragged block at every level."""

    return {
        "schema_version": 1,
        "name": "gpu-scan-test",
        "seed": 78,
        "plan_sha256": plan_sha256(),
        "correctness": {
            "cases": [[31, NORMAL], [1025, INTEGERS], [1048577, NORMAL]],
            "runs": 2,
        },
        "benchmark": {
            "lengths": [4096, 5003],
            "warmup": 1,
            "repeats": 3,
            "target_bytes_per_sample": 1 << 16,
            "max_runtime_seconds": 60,
        },
        "comparison": {"min_faster_ratio": 1.1},
    }


def changed(path, value):
    mapping = copy.deepcopy(make_config_mapping())
    *parents, last = path
    target = mapping
    for key in parents:
        target = target[key]
    target[last] = value
    return mapping


def test_a_short_config_parses_and_counts_its_work():
    config = config_from_mapping(make_config_mapping())

    assert config.to_mapping() == make_config_mapping()
    assert config.correctness.cases == (
        (31, NORMAL),
        (1025, INTEGERS),
        (1048577, NORMAL),
    )
    assert config.planned_work() == PlannedWork(
        correctness_cases=3,
        case_elements=1049633,
        correctness_runs=24,
        benchmark_points=2,
        sides_per_point=5,
        timed_samples=30,
        calls=20 * (2 + 2),
        bytes=20 * (2 * 8 * 4096 + 2 * 8 * 5003),
    )
    assert scan_calls_per_sample(4096, 1 << 16) == 2
    assert scan_calls_per_sample(5003, 1 << 16) == 2
    assert len(config.fingerprint()) == 64
    assert (
        config.fingerprint()
        != config_from_mapping(changed(("seed",), 79)).fingerprint()
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("plan_sha256",), "0" * 64, "does not name the declared plan"),
        (("plan_sha256",), "ABC", "SHA-256 digest"),
        (("name",), "Gpu Scan", "lowercase slug"),
        (("seed",), True, "seed must be an integer"),
        (("schema_version",), 2, "schema_version must be 1"),
        (("correctness", "runs"), 1, "runs must be in"),
        (("correctness", "cases"), [], "1 to 32 cases"),
        (("correctness", "cases"), [[31, NORMAL, 1]], "length, pattern"),
        (("correctness", "cases"), [[31, "uniform"]], "pattern must be one of"),
        (
            ("correctness", "cases"),
            [[31, NORMAL], [31, NORMAL], [1025, INTEGERS], [1048577, NORMAL]],
            "must not repeat",
        ),
        (
            ("correctness", "cases"),
            [[32, NORMAL], [1025, INTEGERS], [1048577, NORMAL]],
            "below 1024 that is not a multiple of 32",
        ),
        (
            ("correctness", "cases"),
            [[31, NORMAL], [1024, INTEGERS], [2048, INTEGERS]],
            "two or more levels that is not a multiple of 1024",
        ),
        (
            ("correctness", "cases"),
            [[31, NORMAL], [1025, INTEGERS]],
            "three levels and a partial block at level 2",
        ),
        (
            ("correctness", "cases"),
            [[31, NORMAL], [1025, NORMAL], [1048577, NORMAL]],
            "a case of each pattern",
        ),
        (
            ("correctness", "cases"),
            [[31, INTEGERS], [1025, NORMAL], [1048577, NORMAL]],
            "an integer case with two or more levels",
        ),
        (
            ("correctness", "cases"),
            [[31, NORMAL], [1025, INTEGERS], [5000003, INTEGERS]],
            "an integer case holds at most",
        ),
        (
            ("correctness", "cases"),
            [[31, NORMAL], [1025, INTEGERS], *([[1 << 24, NORMAL]] * 1)],
            "three levels and a partial block",
        ),
        (("benchmark", "lengths"), [4096, 4096], "must not repeat"),
        (("benchmark", "lengths"), [0], "benchmark length"),
        (("benchmark", "lengths"), list(range(1, 18)), "1 to 16 lengths"),
        (("benchmark", "lengths"), [1], "at most 4096 are allowed"),
        (("benchmark", "repeats"), 2, "repeats must be in"),
        (("benchmark", "warmup"), 11, "warmup must be in"),
        (("benchmark", "target_bytes_per_sample"), 512, "target_bytes_per_sample"),
        (("benchmark", "max_runtime_seconds"), 601, "max_runtime_seconds"),
        (("comparison", "min_faster_ratio"), 1.0, "above 1"),
        (("comparison", "min_faster_ratio"), 1, "finite float"),
        (("benchmark", "extra"), 1, "benchmark fields do not match"),
    ],
)
def test_a_config_that_breaks_a_rule_is_refused(path, value, message):
    with pytest.raises(GpuScanError, match=message):
        config_from_mapping(changed(path, value))


def test_the_case_elements_and_the_run_bytes_are_bounded():
    cases = [[31, NORMAL], [1025, INTEGERS], [1048577, NORMAL]]
    cases += [[(1 << 24) - step, NORMAL] for step in range(2)]
    with pytest.raises(GpuScanError, match=f"the bound is {MAX_CASE_ELEMENTS}"):
        config_from_mapping(changed(("correctness", "cases"), cases))
    lengths = [(1 << 24) - step for step in range(16)]
    mapping = changed(("benchmark", "lengths"), lengths)
    mapping["benchmark"]["target_bytes_per_sample"] = 1 << 30
    mapping["benchmark"]["repeats"] = 50
    mapping["benchmark"]["warmup"] = 10
    with pytest.raises(GpuScanError, match="bytes; the bound is"):
        config_from_mapping(mapping)


def test_config_files_are_read_strictly(tmp_path):
    good = tmp_path / "config.json"
    good.write_text(json.dumps(make_config_mapping()), encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"name": "a", "name": "b"}', encoding="utf-8")
    listing = tmp_path / "list.json"
    listing.write_text("[]", encoding="utf-8")

    assert load_config(good).to_mapping() == make_config_mapping()
    for path in (duplicate, listing, tmp_path / "missing.json", tmp_path):
        with pytest.raises(GpuScanError):
            load_config(path)
