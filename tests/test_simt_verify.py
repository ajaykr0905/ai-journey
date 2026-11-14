from copy import deepcopy
from dataclasses import replace

import pytest

import ai_journey.simt_report as simt_report
from ai_journey.simt_plan import SimtPlan
from ai_journey.simt_report import build_simt_report, seal_simt_report
from ai_journey.simt_verify import SimtVerificationError, verify_simt_report
from test_simt_plan import plan_mapping


def sealed():
    return seal_simt_report(build_simt_report(SimtPlan.from_mapping(plan_mapping())))


def resealed(report):
    return seal_simt_report(report)


def test_verifier_reproduces_every_field_and_returns_the_declared_plan():
    report = sealed()

    plan = verify_simt_report(report)

    assert plan == SimtPlan.from_mapping(plan_mapping())
    assert verify_simt_report(deepcopy(report)) == plan


def test_tampering_is_caught_by_the_seal_before_recomputation():
    report = sealed()
    report["vector_add"]["devices"]["tesla-t4"]["waves"]["waves"] = 2

    with pytest.raises(SimtVerificationError, match="integrity digest"):
        verify_simt_report(report)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("vector_add", "devices", "tesla-t4", "waves", "waves"),
            2,
            "vector_add.devices.tesla-t4.waves.waves does not match",
        ),
        (
            ("vector_add", "devices", "tesla-t4", "waves", "waves"),
            1.0,
            "waves.waves does not match",
        ),
        (
            ("occupancy_sweep", "a100-sxm4-80gb", 1, "occupancy"),
            0.5,
            "occupancy_sweep.a100-sxm4-80gb\\[1\\].occupancy",
        ),
        (
            ("latency_hiding", "memory", 1, "warps_needed"),
            64,
            "latency_hiding.memory\\[1\\].warps_needed",
        ),
        (
            (
                "rgb_to_grayscale",
                "devices",
                "a100-sxm4-80gb",
                "coverage",
                "masked_threads",
            ),
            0,
            "coverage.masked_threads",
        ),
        (
            ("devices", "tesla-t4", "throughput", "ridge_point_flops_per_byte"),
            25.44,
            "ridge_point_flops_per_byte",
        ),
        (("summary", "devices"), 3, "summary.devices"),
    ],
)
def test_resealed_derived_values_are_recomputed_independently(path, value, message):
    report = sealed()
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(SimtVerificationError, match=message):
        verify_simt_report(resealed(report))


def test_resealed_structure_and_boundary_changes_are_rejected():
    report = sealed()
    report["wave_quantization"]["tesla-t4"]["waves"]["measured_seconds"] = 1.0
    with pytest.raises(SimtVerificationError, match="fields do not match"):
        verify_simt_report(resealed(report))

    report = sealed()
    report["latency_hiding"]["memory"].pop()
    with pytest.raises(SimtVerificationError, match="rows do not match"):
        verify_simt_report(resealed(report))

    report = sealed()
    report["evidence_boundary"]["gpu_executed"] = True
    with pytest.raises(SimtVerificationError, match="evidence boundary"):
        verify_simt_report(resealed(report))

    report = sealed()
    report["exclusions"] = report["exclusions"][1:]
    with pytest.raises(SimtVerificationError, match="exclusions"):
        verify_simt_report(resealed(report))

    for field, value in (("measured", 0), ("kind", "measured_simt")):
        report = sealed()
        report[field] = value
        with pytest.raises(SimtVerificationError, match="unmeasured analysis"):
            verify_simt_report(resealed(report))

    report = sealed()
    report["schema_version"] = True
    with pytest.raises(SimtVerificationError, match="schema_version"):
        verify_simt_report(resealed(report))

    report = sealed()
    report["notes"] = "extra"
    with pytest.raises(SimtVerificationError, match="fields do not match the schema"):
        verify_simt_report(resealed(report))


def test_declared_inputs_must_still_match_the_pinned_catalog():
    report = sealed()
    report["declared_inputs"]["devices"][1]["dram_bandwidth_gb_per_second"] = 2_000

    with pytest.raises(SimtVerificationError, match="declared inputs are invalid"):
        verify_simt_report(resealed(report))
    with pytest.raises(SimtVerificationError, match="fields do not match the schema"):
        verify_simt_report([("schema_version", 1)])


def test_independent_arithmetic_catches_a_defect_in_the_analysis_modules(
    monkeypatch,
):
    # Simulate a builder bug: one wave too few. The digest is valid and the
    # buggy builder would reproduce itself, but the verifier's own arithmetic
    # does not.
    def short_waves(device, *, total_blocks, active_blocks_per_sm):
        waves = simt_report_wave_quantization(
            device,
            total_blocks=total_blocks,
            active_blocks_per_sm=active_blocks_per_sm,
        )
        return replace(waves, total_blocks=max(1, total_blocks - 1))

    simt_report_wave_quantization = simt_report.wave_quantization
    monkeypatch.setattr(simt_report, "wave_quantization", short_waves)
    report = sealed()

    with pytest.raises(
        SimtVerificationError,
        match="vector_add.devices.a100-sxm4-80gb.waves.last_wave_fill does not match",
    ):
        verify_simt_report(report)
