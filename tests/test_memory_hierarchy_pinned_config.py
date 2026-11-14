import itertools
import json
from pathlib import Path

from ai_journey.memory_hierarchy_config import MAX_ALLOCATION_BYTES, load_config
from ai_journey.memory_hierarchy_report import measure_report
from ai_journey.memory_hierarchy_stream import stream_kernel
from ai_journey.memory_hierarchy_verify import verify_report

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-71-memory-hierarchy.json"
MIB = 1 << 20


def test_pinned_controls_parse_to_the_checked_in_mapping():
    config = load_config(PINNED)

    assert config.to_mapping() == json.loads(PINNED.read_text(encoding="utf-8"))
    assert config.name == "day-71-memory-hierarchy"
    assert (config.dtype, config.seed) == ("float64", 71)
    assert config.timing.to_mapping() == {
        "warmup": 2,
        "repeats": 11,
        "target_bytes_per_sample": 32 * MIB,
        "max_runtime_seconds": 20,
    }
    assert config.stream.kernels == ("copy", "scale", "add", "triad")
    assert (config.matmul.m, config.matmul.n, config.matmul.k) == (1024, 1024, 1024)
    note = config.littles_law.latency_note
    assert config.littles_law.latency_ns == 100
    assert "does not measure latency" in note
    assert "not a datasheet value for this host" in note


def test_pinned_sweep_runs_from_cache_sized_to_far_beyond_cache_sized_sets():
    config = load_config(PINNED)
    triad = stream_kernel("triad")
    sizes = [
        triad.working_set_bytes(count, "float64") for count in config.stream.elements
    ]

    assert sizes[0] == 24 * 1024
    assert sizes[-1] == 192 * MIB
    assert len(sizes) == 12 and sizes == sorted(set(sizes))
    assert config.access.elements * 8 == 64 * MIB
    assert config.access.stride_elements * 8 == 128


def test_pinned_work_fits_the_memory_bound_and_documented_budget():
    work = load_config(PINNED).planned_work()

    assert work.to_mapping() == {
        "peak_allocation_bytes": 208 * MIB,
        "accounted_bytes": 45_725_335_552,
        "calls": 109_525,
        "flops": 29_377_107_968,
        "measurements": 53,
    }
    assert work.peak_allocation_bytes <= MAX_ALLOCATION_BYTES


def test_the_pinned_run_completes_its_checks_and_verifies():
    config = load_config(PINNED)

    # A scripted clock keeps the budget deterministic; every kernel and check
    # still runs at full size.
    report = measure_report(config, clock=itertools.count(0, 1_000).__next__)

    summary = verify_report(report, config=config)
    assert (summary["stream_measurements"], summary["access_patterns"]) == (48, 4)
    assert summary["timed_samples"] == 53 * 11
    assert report["planned_work"] == config.planned_work().to_mapping()
    assert report["matmul"]["correctness"]["checked_elements"] == 1024 * 1024
