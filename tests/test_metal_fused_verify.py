import copy
import dataclasses
import time

import pytest

from ai_journey.memory_access_report import report_digest
from ai_journey.metal_fused_config import config_from_mapping
from ai_journey.metal_fused_report import measure_report
from ai_journey.metal_fused_source import F1, F2, F3, U0
from ai_journey.metal_fused_verify import MetalFusedVerificationError, verify_report
from test_kernel_metal_runtime import requires_metal
from test_metal_fused_ceiling import recorded_ceiling
from test_metal_fused_config import make_config_mapping
from test_metal_fused_report import small_config, synthetic_report


def reseal(report):
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    return {**unsigned, "report_sha256": report_digest(unsigned)}


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    return synthetic_report(tmp_path_factory.mktemp("verify"))


def test_a_stand_in_report_verifies_as_synthetic_timing(sealed, tmp_path):
    summary = verify_report(
        sealed, config=small_config(), ceiling=recorded_ceiling(tmp_path)
    )

    assert summary["measured"] is False
    assert summary["scope"] == "synthetic_backend_or_clock"
    assert summary["report_sha256"] == sealed["report_sha256"]
    assert summary["correctness_cases"] == 2 and summary["benchmark_points"] == 2
    assert summary["refused_steps"] == []
    assert summary["ceiling_report_sha256"] == recorded_ceiling(tmp_path).report_sha256
    assert set(summary["hypotheses"]) == {U0, F1, F2, F3}


def test_the_report_binds_to_its_own_config_and_ceiling(sealed, tmp_path):
    other = config_from_mapping(make_config_mapping(seed=80))
    with pytest.raises(MetalFusedVerificationError, match="supplied config"):
        verify_report(sealed, config=other)
    ceiling = dataclasses.replace(recorded_ceiling(tmp_path), gb_per_second=90.0)
    with pytest.raises(MetalFusedVerificationError, match="supplied ceiling"):
        verify_report(sealed, ceiling=ceiling)
    with pytest.raises(TypeError):
        verify_report(sealed, config={"name": "x"})


def _sample(report):
    report["benchmark"]["points"][0]["sides"][F1]["samples_ns"][0] += 1


def _verdict(report):
    row = report["speedup"]["points"][0]["rows"][1]
    row["held"] = not row["held"]


def _measured(report):
    report["measured"] = True


def _boundary(report):
    report["evidence_boundary"]["nvidia_gpu_used"] = True


def _hypothesis(report):
    report["plan"]["steps"][2]["hypothesis"] = "F2 is always faster."


def _ratio(report):
    report["correctness"]["cases"][0]["outputs"][2]["max_bound_ratio"] = 1.5


def _bits(report):
    report["correctness"]["cases"][0]["outputs"][3][
        "differing_from_rounded_reference"
    ] = 0


def _bound(report):
    report["correctness"]["cases"][1]["outputs"][0]["bound_sha256"] = "0" * 64


def _inputs(report):
    report["correctness"]["cases"][0]["input_sha256"] = "1" * 64


def _naive(report):
    report["correctness"]["cases"][1]["naive_float32_finite"] = True


def _limits(report):
    report["shaders"]["kernels"][5]["limits"]["max_threads_per_threadgroup"] = 512


def _same(report):
    report["correctness"]["cases"][0]["same_bits"][0]["same"] = False


def _ceiling_rate(report):
    report["ceiling"]["gb_per_second"] = 120.0


def _probe(report):
    report["exp_probe"]["max_relative_error"] = 1.0e-5


def _planned(report):
    report["planned_work"]["calls"] += 1


def _elapsed(report):
    report["run"]["elapsed_ns"] = report["run"]["timed_ns"] - 1


def _summary(report):
    report["summary"]["max_bound_ratio"] = 0.0


def _point_check(report):
    report["benchmark"]["points"][1]["checks"]["torch_softmax"]["max_bound_ratio"] = 2.0


def _model(report):
    report["benchmark"]["points"][0]["sides"][U0]["modeled_bytes_per_call"] += 8


def _fraction(report):
    rate = report["speedup"]["points"][0]["rows"][0]["rate"]
    rate["effective_fraction_of_ceiling"] *= 2


def _warnings(report):
    report["warnings"]["count"] = 3


def _switches(report):
    report["torch_mps_switches"]["values"]["PYTORCH_MPS_FAST_MATH"] = "on/off"


def _extra(report):
    report["notes"] = "faster than cuDNN"


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (_sample, "point 0 does not recompute"),
        (_verdict, "speedup rows or the hypotheses"),
        (_measured, "only the host timer"),
        (_boundary, "boundary were weakened"),
        (_hypothesis, "declared plan"),
        (_ratio, "breaks a correctness rule"),
        (_bits, "claims the rounded reference bits"),
        (_bound, "case 1 does not rebuild"),
        (_inputs, "case 0 does not rebuild"),
        (_naive, "case 1 does not rebuild"),
        (_limits, "shader section does not recompute"),
        (_same, "case 0 does not rebuild"),
        (_ceiling_rate, "speedup rows"),
        (_probe, "assumption of the bounds"),
        (_planned, "planned work"),
        (_elapsed, "run time does not cover"),
        (_summary, "summary does not reconcile"),
        (_point_check, "breaks its bound"),
        (_model, "point 0 does not recompute"),
        (_fraction, "speedup rows"),
        (_warnings, "warning counts"),
        (_switches, "MPS switches"),
        (_extra, "fields do not match"),
    ],
)
def test_each_tampered_section_is_rejected_after_a_fresh_seal(sealed, tamper, message):
    report = copy.deepcopy(sealed)
    tamper(report)

    with pytest.raises(MetalFusedVerificationError, match=message):
        verify_report(reseal(report))


def test_an_unsealed_change_is_named_as_a_digest_mismatch(sealed):
    report = copy.deepcopy(sealed)
    report["summary"]["warnings"] = 9

    with pytest.raises(MetalFusedVerificationError, match="digest does not match"):
        verify_report(report)
    with pytest.raises(MetalFusedVerificationError, match="JSON object"):
        verify_report([sealed])


@requires_metal
def test_a_report_from_this_gpu_verifies_as_a_measurement(tmp_path):
    ceiling = recorded_ceiling(tmp_path)
    report = measure_report(small_config(), ceiling, clock=time.perf_counter_ns)

    summary = verify_report(report, config=small_config(), ceiling=ceiling)

    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_this_host"
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
