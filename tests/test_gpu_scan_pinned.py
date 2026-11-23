import json
from pathlib import Path

from ai_journey.gpu_scan_config import (
    MAX_RUN_BYTES,
    MAX_RUN_CALLS,
    PlannedWork,
    load_config,
    scan_calls_per_sample,
)
from ai_journey.gpu_scan_plan import LENGTHS, plan_sha256
from ai_journey.gpu_scan_source import level_lengths

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-78-gpu-scan.json"


def test_the_pinned_config_times_every_length_that_the_plan_predicts():
    config = load_config(PINNED)

    assert config.name == "day-78-gpu-scan" and config.seed == 78
    assert config.plan_sha256 == plan_sha256()
    assert config.benchmark.lengths == LENGTHS
    assert config.benchmark.target_bytes_per_sample == 1 << 30
    assert config.comparison.min_faster_ratio == 1.1
    assert config.to_mapping() == json.loads(PINNED.read_text(encoding="utf-8"))


def test_the_pinned_cases_check_both_patterns_and_a_ragged_block_at_each_level():
    cases = load_config(PINNED).correctness.cases

    assert [length for length, _ in cases] == [
        1,
        31,
        32,
        1000,
        1024,
        1025,
        4097,
        65536,
        1048577,
        1048577,
        5000003,
    ]
    assert {pattern for _, pattern in cases} == {"standard_normal", "small_integers"}
    assert level_lengths(1048577) == (1048577, 1025, 2)
    assert level_lengths(5000003) == (5000003, 4883, 5)


def test_the_pinned_run_is_counted_and_fits_every_work_bound():
    config = load_config(PINNED)
    work = config.planned_work()

    assert [scan_calls_per_sample(length, 1 << 30) for length in LENGTHS] == [
        2048,
        128,
        8,
        27,
    ]
    assert work == PlannedWork(
        correctness_cases=11,
        case_elements=7169903,
        correctness_runs=132,
        benchmark_points=4,
        sides_per_point=5,
        timed_samples=220,
        calls=143715,
        bytes=279579697800,
    )
    assert work.calls <= MAX_RUN_CALLS and work.bytes <= MAX_RUN_BYTES
