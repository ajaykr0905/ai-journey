from copy import deepcopy

import pytest

import ai_journey.memory_access_report as report_module
from ai_journey.memory_access_banks import analyze_banks
from ai_journey.memory_access_baseline import cpu_environment
from ai_journey.memory_access_coalescing import CoalescingAnalysis, analyze_coalescing
from ai_journey.memory_access_pattern import broadcast_access
from ai_journey.memory_access_plan import plan_from_mapping
from ai_journey.memory_access_report import (
    build_analysis_report,
    build_measurement_report,
    seal_report,
)
from ai_journey.memory_access_verify import (
    MemoryAccessVerificationError,
    verify_report,
)
from test_memory_access_plan import make_plan_mapping
from test_memory_access_report import make_measurements, make_plan


def reseal(report):
    return seal_report(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )


def analysis():
    return seal_report(build_analysis_report(make_plan()))


def measurement():
    plan = make_plan()
    return seal_report(
        build_measurement_report(plan, make_measurements(plan), cpu_environment())
    )


def other_plan():
    mapping = make_plan_mapping()
    mapping["name"] = "another-plan"
    return plan_from_mapping(mapping)


def test_sealed_analysis_survives_independent_recomputation_and_plan_binding():
    report = analysis()
    summary = verify_report(report, plan=make_plan())

    assert summary == {
        "kind": "memory_access_analysis",
        "report_sha256": report["report_sha256"],
        "plan_sha256": make_plan().fingerprint(),
        "patterns": 9,
        "transposes": 2,
        "tiled_estimates": 4,
        "measured": False,
    }
    with pytest.raises(MemoryAccessVerificationError, match="supplied plan"):
        verify_report(report, plan=other_plan())


def tamper(path, change):
    def apply(report):
        target = report
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = change(target[path[-1]])
        return report

    return apply


def insert(path, value):
    def apply(report):
        target = report
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return report

    return apply


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            tamper(("patterns", 0, "global_memory", "sectors"), lambda v: v + 1),
            "global accounting",
        ),
        (
            tamper(("patterns", 2, "shared_memory", "conflict_degree"), lambda v: 1),
            "bank accounting",
        ),
        (
            tamper(("patterns", 8, "shared_memory"), lambda v: {"conflict_degree": 1}),
            "bank accounting",
        ),
        (
            tamper(("patterns", 0, "access", "active_mask"), lambda v: 1),
            "lane mask",
        ),
        (
            tamper(("patterns", 0, "access", "lane_addresses"), lambda v: [4] * 32),
            "global accounting",
        ),
        (
            tamper(
                ("transposes", 0, "shared_column_reads_per_tile", "wavefronts"),
                lambda v: 32,
            ),
            "transpose unpadded",
        ),
        (
            tamper(
                ("matmul", "sweeps", 0, "tiled", 0, "traffic", "useful_loads"),
                lambda v: v - 1,
            ),
            "matmul sweep 64x64x64",
        ),
        (
            tamper(
                ("matmul", "sweeps", 1, "naive", "roofline", "bottleneck"),
                lambda v: "compute",
            ),
            "matmul sweep 10x6x7",
        ),
        (
            tamper(("matmul", "inner_loop_banks", 1, "warps"), lambda v: v + 1),
            "inner-loop",
        ),
        (
            tamper(("matmul", "roofline", "ridge_point_flops_per_byte"), lambda v: 5.0),
            "ridge point",
        ),
        (
            tamper(("summary", "patterns"), lambda v: v + 1),
            "summary does not reconcile",
        ),
        (tamper(("exclusions",), lambda v: v[:-1]), "rebuilt plan"),
        (tamper(("model_sources",), lambda v: []), "rebuilt plan"),
        (
            tamper(("evidence_boundary", "gpu_executed"), lambda v: True),
            "boundary was weakened",
        ),
        (tamper(("measured",), lambda v: True), "must remain unmeasured"),
        (tamper(("plan_sha256",), lambda v: "0" * 64), "plan digest mismatch"),
        (tamper(("plan", "geometry", "warp_size"), lambda v: 64), "documented CUDA"),
        (tamper(("schema_version",), lambda v: True), "unsupported report schema"),
        (tamper(("patterns",), lambda v: v[:-1]), "do not cover the plan"),
    ],
)
def test_resealed_analysis_drift_is_caught_by_recomputation(change, message):
    report = reseal(change(deepcopy(analysis())))

    with pytest.raises(MemoryAccessVerificationError, match=message):
        verify_report(report)


def test_consistently_forged_lane_addresses_fail_the_plan_rebuild():
    report = deepcopy(analysis())
    forged = broadcast_access(address_bytes=4)
    report["patterns"][0]["access"] = forged.to_mapping()
    report["patterns"][0]["global_memory"] = analyze_coalescing(forged).to_mapping()
    report["patterns"][0]["shared_memory"] = analyze_banks(forged).to_mapping()

    with pytest.raises(MemoryAccessVerificationError, match="rebuilt plan"):
        verify_report(reseal(report))


def test_unsealed_edits_and_unknown_fields_are_rejected():
    report = analysis()
    edited = deepcopy(report)
    edited["summary"]["patterns"] += 1
    extra = reseal({**report, "gpu_seconds": 1.0})

    with pytest.raises(MemoryAccessVerificationError, match="digest"):
        verify_report(edited)
    with pytest.raises(MemoryAccessVerificationError, match="fields do not match"):
        verify_report(extra)
    with pytest.raises(MemoryAccessVerificationError, match="JSON object"):
        verify_report([report])


def test_first_principles_checks_catch_a_model_bug_that_rebuilding_would_repeat(
    monkeypatch,
):
    real = report_module.analyze_coalescing

    def buggy(access):
        result = real(access)
        return CoalescingAnalysis(
            result.access_bytes,
            result.active_lanes,
            result.sectors[:1],
            result.lines,
            result.requested_bytes,
            result.unique_bytes,
        )

    monkeypatch.setattr(report_module, "analyze_coalescing", buggy)
    report = seal_report(build_analysis_report(make_plan()))

    with pytest.raises(MemoryAccessVerificationError, match="global accounting"):
        verify_report(report)


def test_sealed_measurement_passes_structural_and_accounting_checks():
    report = measurement()
    summary = verify_report(report, plan=make_plan())

    assert summary == {
        "kind": "memory_access_cpu_measurement",
        "report_sha256": report["report_sha256"],
        "plan_sha256": make_plan().fingerprint(),
        "cases": 4,
        "measured": True,
        "scope": "cpu_wall_clock_this_host",
    }
    with pytest.raises(MemoryAccessVerificationError, match="supplied plan"):
        verify_report(report, plan=other_plan())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            tamper(("results", 0, "samples_ns"), lambda v: [v[0] + 1] + v[1:]),
            "timing statistics",
        ),
        (
            tamper(("results", 0, "samples_ns"), lambda v: v[:-1]),
            "one positive integer per repeat",
        ),
        (
            tamper(("results", 0, "samples_ns"), lambda v: [0] + v[1:]),
            "one positive integer per repeat",
        ),
        (
            tamper(("results", 1, "median_gflops_per_second"), lambda v: v * 2),
            "GFLOP/s",
        ),
        (tamper(("results", 1, "flops"), lambda v: v + 1), "GFLOP/s"),
        (
            tamper(("results", 2, "correctness", "max_bound_ratio"), lambda v: 1.5),
            "rounding bound",
        ),
        (
            tamper(("results", 2, "correctness", "tolerance"), lambda v: v * 10),
            "rounding bound",
        ),
        (
            tamper(("results", 2, "correctness", "checked_results"), lambda v: v - 1),
            "rounding bound",
        ),
        (tamper(("results", 3, "device"), lambda v: "cuda"), "planned case"),
        (tamper(("results", 3, "backend"), lambda v: "numpy"), "planned case"),
        (
            tamper(("results", 3, "input_sha256"), lambda v: "f" * 64),
            "different inputs",
        ),
        (tamper(("results", 3, "input_sha256"), lambda v: "not-a-digest"), "SHA-256"),
        (tamper(("results",), lambda v: v[:-1]), "cover the planned cases"),
        (
            tamper(("evidence_boundary", "gpu_executed"), lambda v: True),
            "boundary was weakened",
        ),
        (tamper(("measurement_scope",), lambda v: "gpu"), "boundary was weakened"),
        (tamper(("measured",), lambda v: False), "must be measured"),
        (tamper(("timer",), lambda v: "time.time"), "method fields"),
        (insert(("environment", "hostname"), "lab-host"), "environment fields"),
        (tamper(("environment", "cpu_count"), lambda v: 0), "cpu_count"),
        (tamper(("environment", "torch_threads"), lambda v: True), "torch_threads"),
        (tamper(("environment", "platform_system"), lambda v: ""), "platform token"),
        (
            tamper(("environment", "platform_machine"), lambda v: "/home/lab/x86_64"),
            "platform token",
        ),
        (
            tamper(("environment", "python_version"), lambda v: "3.11 on lab-host"),
            "platform token",
        ),
        (tamper(("plan_name",), lambda v: "Lab Host Plan"), "plan slug"),
        (tamper(("plan_name",), lambda v: "/srv/plans/day-74"), "plan slug"),
        (tamper(("benchmark", "repeats"), lambda v: v + 1), "planned case"),
        (tamper(("benchmark", "backends"), lambda v: ["cupy"]), "failed verification"),
        (tamper(("summary", "checked_results"), lambda v: v + 1), "summary"),
        (tamper(("plan_sha256",), lambda v: "plan"), "SHA-256"),
    ],
)
def test_resealed_measurement_drift_is_rejected(change, message):
    report = reseal(change(deepcopy(measurement())))

    with pytest.raises(MemoryAccessVerificationError, match=message):
        verify_report(report)
