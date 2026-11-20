import copy
import functools
import itertools

import pytest

from ai_journey.gpu_ceiling_source import GpuCeilingError
from ai_journey.gpu_scan_config import config_from_mapping
from ai_journey.gpu_scan_plan import COPY, VENDOR, plan_mapping, plan_sha256
from ai_journey.gpu_scan_report import (
    MEASUREMENT_BOUNDARY,
    REPORT_KIND,
    SYNTHETIC_BOUNDARY,
    build_report,
    measure_report,
)
from ai_journey.gpu_scan_source import (
    KERNELS,
    S1_BLELLOCH,
    S2_SIMD,
    GpuScanError,
    library_sha256,
)
from ai_journey.gpu_scan_verify import verify_report
from ai_journey.kernel_metal_benchmark import KernelMetalError
from ai_journey.memory_access_report import report_digest
from test_gpu_scan_config import make_config_mapping
from test_gpu_scan_correctness import StandInBackend
from test_kernel_metal_runtime import requires_metal

LOAD = (1.5, 2.25, 3.0)


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


def stand_in_report(mapping=None, **faults):
    return measure_report(
        config_from_mapping(mapping or make_config_mapping()),
        backend=StandInBackend(**faults),
        clock=scripted_clock(),
        load_average=lambda: LOAD,
    )


@functools.cache
def _cached():
    return stand_in_report()


def sealed():
    """A fresh copy of one stand-in report, built once per test process."""

    return copy.deepcopy(_cached())


def test_a_stand_in_run_seals_a_synthetic_report_with_the_declared_plan():
    report = sealed()

    assert report["kind"] == REPORT_KIND
    assert report["report_sha256"] == report_digest(report)
    assert (report["measured"], report["backend"], report["timer"]) == (
        False,
        "injected",
        "injected",
    )
    assert report["label"] == "synthetic timing, not a measurement"
    assert report["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert report["plan"] == {**plan_mapping(), "sha256": plan_sha256()}
    assert report["config"]["plan_sha256"] == plan_sha256()
    assert report["shaders"]["library_sha256"] == library_sha256()
    assert report["host_load"]["before"] == list(LOAD)
    assert report["refusals"] == []
    assert report["correctness"]["runs"] == 2
    assert [point["length"] for point in report["benchmark"]["points"]] == [4096, 5003]
    summary = report["summary"]
    assert summary["kernels_checked"] == list(KERNELS)
    assert summary["cases_with_other_bits"] == {
        "scan_s0_hillis_steele": [],
        "scan_s1_blelloch": [],
    }
    assert summary["nondeterministic_cases"] == {
        name: [] for name in (*KERNELS, VENDOR)
    }
    assert summary["vendor_missed_exact_at"] == []
    assert summary["timed_samples"] == 2 * 5 * 3
    assert summary["hypotheses"][S2_SIMD] == {
        "held": 0,
        "missed": 0,
        "no_verdict": 0,
        "no_prediction": 2,
    }
    assert summary["max_bound_ratio"] < 0.2
    assert [entry["length"] for entry in summary["points"]] == [4096, 5003]
    assert verify_report(report)["measured"] is False


def test_the_measured_boundary_rules_out_every_claim_the_lab_cannot_make():
    false = [name for name, value in MEASUREMENT_BOUNDARY.items() if value is False]

    assert MEASUREMENT_BOUNDARY["apple_gpu_metal_this_host"] is True
    assert {
        "cuda_kernel_executed",
        "nvidia_gpu_used",
        "warp_shuffle_executed",
        "vendor_kernel_known",
        "hardware_counters_used",
        "occupancy_measured",
        "production_claim",
    } <= set(false)
    statement = MEASUREMENT_BOUNDARY["statement"]
    assert "not the same instruction" in statement
    assert "a bound name is an inference" in statement
    assert SYNTHETIC_BOUNDARY["synthetic_timing"] is True


def test_a_refused_kernel_and_a_measured_bit_change_are_reported_not_hidden():
    report = stand_in_report(widths={S2_SIMD: 16}, wobble={VENDOR})

    assert [entry["name"] for entry in report["refusals"]] == [S2_SIMD]
    assert report["summary"]["refused_kernels"] == [S2_SIMD]
    assert report["summary"]["kernels_checked"] == list(KERNELS)[:2]
    assert report["summary"]["nondeterministic_cases"][VENDOR] == [0, 1, 2]
    rows = report["speedup"]["points"][0]["rows"]
    assert [row["ran"] for row in rows] == [True, True, False]
    assert verify_report(report)["refused_kernels"] == [S2_SIMD]


def test_a_run_past_its_budget_or_with_a_wrong_output_writes_no_report():
    mapping = make_config_mapping()
    mapping["benchmark"]["max_runtime_seconds"] = 1
    with pytest.raises(KernelMetalError, match="exceeded max_runtime_seconds=1"):
        measure_report(
            config_from_mapping(mapping),
            backend=StandInBackend(),
            clock=scripted_clock(step=10**9),
            load_average=lambda: LOAD,
        )
    with pytest.raises(GpuScanError, match="wrote into its guard tail"):
        stand_in_report(leak={S1_BLELLOCH})
    with pytest.raises(TypeError, match="GpuScanConfig"):
        build_report(make_config_mapping(), backend=StandInBackend())
    with pytest.raises(GpuCeilingError, match="load average"):
        measure_report(
            config_from_mapping(make_config_mapping()),
            backend=StandInBackend(),
            clock=scripted_clock(),
            load_average=lambda: (1.0, 2.0),
        )


def test_the_copy_and_add_kernels_are_named_with_their_limits():
    shaders = sealed()["shaders"]

    assert shaders["copy"]["kernel"] == "stream_copy"
    assert shaders["copy"]["group_size"] == 256
    assert shaders["copy"]["float4_per_thread"] == 4
    assert shaders["add"]["name"] == "scan_add_offsets"
    assert [entry["name"] for entry in shaders["kernels"]] == list(KERNELS)
    assert all(entry["admission"]["admitted"] for entry in shaders["kernels"])
    assert set(sealed()["benchmark"]["points"][0]["sides"]) == {
        *KERNELS,
        VENDOR,
        COPY,
    }


@requires_metal
def test_this_gpu_makes_a_short_report_that_verifies_as_a_measurement():
    report = measure_report(config_from_mapping(make_config_mapping()))

    summary = verify_report(report)
    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_this_host"
    assert report["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert summary["cases_with_other_bits"] == {
        "scan_s0_hillis_steele": [],
        "scan_s1_blelloch": [],
    }
    assert summary["max_bound_ratio"] <= 1.0
