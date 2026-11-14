import itertools
import json
from copy import deepcopy

import pytest

from ai_journey.memory_access_baseline import cpu_environment, measure_matmul
from ai_journey.memory_access_pattern import MemoryAccessError
from ai_journey.memory_access_plan import canonical_json_bytes, plan_from_mapping
from ai_journey.memory_access_report import (
    ANALYSIS_BOUNDARY,
    ANALYSIS_KIND,
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_KIND,
    build_analysis_report,
    build_measurement_report,
    report_digest,
    seal_report,
)
from test_memory_access_plan import make_plan_mapping


def make_plan():
    return plan_from_mapping(make_plan_mapping())


def make_measurements(plan, step=500):
    clock = itertools.count(10_000, step).__next__
    benchmark = plan.benchmark
    return tuple(
        measure_matmul(
            case,
            warmup=benchmark.warmup,
            repeats=benchmark.repeats,
            seed=benchmark.seed,
            clock=clock,
        )
        for case in benchmark.cases()
    )


def test_analysis_report_recomputes_patterns_transposes_and_tile_sweeps():
    plan = make_plan()
    report = build_analysis_report(plan)
    patterns = {item["name"]: item for item in report["patterns"]}

    assert report["kind"] == ANALYSIS_KIND
    assert report["measured"] is False
    assert report["evidence_boundary"] == ANALYSIS_BOUNDARY
    assert report["plan"] == plan.to_mapping()
    assert report["plan_sha256"] == plan.fingerprint()
    assert patterns["contiguous"]["global_memory"]["sectors"] == 4
    assert patterns["offset"]["global_memory"]["sectors"] == 5
    assert patterns["stride-2"]["shared_memory"]["conflict_degree"] == 2
    assert patterns["broadcast"]["global_memory"]["efficiency"] == 0.125
    assert patterns["half-warp"]["global_memory"]["active_lanes"] == 16
    assert patterns["float64"]["shared_memory"] is None
    assert patterns["float64"]["global_memory"]["sectors"] == 8
    transposes = {item["name"]: item for item in report["transposes"]}
    unpadded = transposes["unpadded"]["shared_column_reads_per_tile"]
    assert unpadded["max_conflict_degree"] == 32
    padded = transposes["padded"]["shared_column_reads_per_tile"]
    assert padded["max_conflict_degree"] == 1


def test_analysis_report_classifies_every_tile_on_the_declared_roofline():
    report = build_analysis_report(make_plan())
    matmul = report["matmul"]
    first = matmul["sweeps"][0]

    assert matmul["roofline"]["ridge_point_flops_per_byte"] == 4
    assert matmul["roofline"]["declared"] is True
    assert [item["tile"] for item in matmul["inner_loop_banks"]] == [4, 8]
    assert first["shape"] == {"m": 64, "n": 64, "k": 64}
    assert first["naive"]["roofline"]["bottleneck"] == "memory"
    assert first["compulsory"]["bottleneck"] == "compute"
    assert [item["traffic"]["tile"] for item in first["tiled"]] == [4, 8]
    assert first["tiled"][1]["traffic"]["useful_loads"] == 2 * 64**3 // 8
    assert report["summary"] == {
        "patterns": 9,
        "shared_memory_patterns": 8,
        "transposes": 2,
        "matmul_shapes": 2,
        "tiles": 2,
        "tiled_estimates": 4,
        "tiled_bottleneck_counts": {"compute": 0, "memory": 4, "ridge": 0},
    }


def test_analysis_report_is_canonical_json_and_rebuilds_identically():
    report = build_analysis_report(make_plan())

    assert json.loads(canonical_json_bytes(report)) == report
    assert build_analysis_report(make_plan()) == report
    assert len(canonical_json_bytes(report)) < 256 * 1024


def test_seal_is_deterministic_detects_drift_and_does_not_mutate():
    report = build_analysis_report(make_plan())
    sealed = seal_report(report)

    assert "report_sha256" not in report
    assert sealed == seal_report(report)
    assert sealed["report_sha256"] == report_digest(sealed) == report_digest(report)
    drifted = deepcopy(report)
    drifted["patterns"][0]["global_memory"]["sectors"] += 1
    assert report_digest(drifted) != sealed["report_sha256"]
    with pytest.raises(MemoryAccessError, match="already sealed"):
        seal_report(sealed)
    with pytest.raises(TypeError, match="dictionary"):
        seal_report([])


def test_measurement_report_records_cpu_samples_for_every_planned_case():
    plan = make_plan()
    measurements = make_measurements(plan)
    report = build_measurement_report(plan, measurements, cpu_environment())

    assert report["kind"] == MEASUREMENT_KIND
    assert report["measured"] is True
    assert report["measurement_scope"] == "cpu_wall_clock_this_host"
    assert report["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert report["plan_sha256"] == plan.fingerprint()
    assert report["benchmark"] == plan.benchmark.to_mapping()
    assert [
        (item["backend"], item["m"], item["n"], item["k"]) for item in report["results"]
    ] == [
        ("numpy", 8, 8, 8),
        ("numpy", 16, 12, 8),
        ("torch", 8, 8, 8),
        ("torch", 16, 12, 8),
    ]
    assert all(item["device"] == "cpu" for item in report["results"])
    assert all(item["samples_ns"] == [500] * 3 for item in report["results"])
    assert report["summary"]["cases"] == 4
    assert report["summary"]["checked_results"] == 16
    assert report["summary"]["max_bound_ratio"] < 1
    assert json.loads(canonical_json_bytes(report)) == report


def test_measurement_report_refuses_evidence_that_does_not_match_the_plan():
    plan = make_plan()
    measurements = make_measurements(plan)
    other = plan_from_mapping(
        {
            **make_plan_mapping(),
            "benchmark": {**make_plan_mapping()["benchmark"], "repeats": 4},
        }
    )

    with pytest.raises(MemoryAccessError, match="cases in order"):
        build_measurement_report(plan, measurements[::-1], cpu_environment())
    with pytest.raises(MemoryAccessError, match="cases in order"):
        build_measurement_report(plan, measurements[:2], cpu_environment())
    with pytest.raises(MemoryAccessError, match="benchmark budget"):
        build_measurement_report(other, measurements, cpu_environment())
    with pytest.raises(MemoryAccessError, match="environment"):
        build_measurement_report(
            plan, measurements, {**cpu_environment(), "hostname": "x"}
        )
    with pytest.raises(TypeError, match="MatmulMeasurement"):
        build_measurement_report(plan, list(measurements), cpu_environment())
    with pytest.raises(TypeError, match="MemoryAccessPlan"):
        build_analysis_report(make_plan_mapping())
