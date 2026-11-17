import json

import pytest

from ai_journey.gpu_ceiling_roofline import (
    BANDWIDTH_BOUND,
    COMPUTE_BOUND,
    MeasuredRoofline,
    load_roofline,
    roofline_from_report,
)
from ai_journey.gpu_ceiling_source import GpuCeilingError
from ai_journey.gpu_ceiling_verify import GpuCeilingVerificationError
from test_gpu_ceiling_recorded import RECORDED, RECORDED_REPORT_JSON
from test_gpu_ceiling_verify import SYNTHETIC, fresh, reseal

DIGEST = "a" * 64


def roofline(compute=4618.0, bandwidth=100.7, measured=True):
    label = (
        "host-specific Apple GPU (Metal) measurement"
        if measured
        else "synthetic timing, not a measurement"
    )
    return MeasuredRoofline(compute, bandwidth, DIGEST, measured, label)


def test_a_verified_measured_report_gives_its_two_ceilings():
    measured = roofline_from_report(json.loads(RECORDED_REPORT_JSON))
    summary = RECORDED["summary"]

    assert (
        measured.compute_gflops_per_second == summary["fma_ceiling_gflops_per_second"]
    )
    assert (
        measured.bandwidth_gb_per_second == summary["bandwidth_ceiling_gb_per_second"]
    )
    assert measured.ridge_flops_per_byte == summary["ridge_point_flops_per_byte"]
    assert (measured.measured, measured.report_sha256) == (
        True,
        RECORDED["report_sha256"],
    )
    assert measured.to_mapping()["label"] == summary["label"]


def test_the_attainable_rate_is_the_lower_ceiling_at_each_intensity():
    line = roofline()

    assert line.attainable_gflops_per_second(8) == 100.7 * 8
    assert line.bound(8) == BANDWIDTH_BOUND
    assert line.attainable_gflops_per_second(46) == 4618.0
    assert line.bound(46) == COMPUTE_BOUND
    ridge = line.ridge_flops_per_byte
    assert ridge == 4618.0 / 100.7
    assert line.bound(ridge * 1.001) == COMPUTE_BOUND
    assert line.bound(ridge * 0.999) == BANDWIDTH_BOUND


def test_below_ceiling_reports_the_fraction_and_keeps_the_labels():
    comparison = roofline().below_ceiling(403.0, 8)

    assert comparison == {
        "achieved_gflops_per_second": 403.0,
        "flops_per_byte": 8.0,
        "attainable_gflops_per_second": 100.7 * 8,
        "fraction_of_ceiling": 403.0 / (100.7 * 8),
        "bound": "bandwidth",
        "ceiling_report_sha256": DIGEST,
        "measured": True,
        "label": "host-specific Apple GPU (Metal) measurement",
    }
    # These ceilings are the best of this lab's kernels, so a kernel can pass them.
    assert roofline().below_ceiling(5000.0, 64)["fraction_of_ceiling"] > 1


def test_a_synthetic_report_gives_a_roofline_that_says_so():
    synthetic = roofline_from_report(fresh())

    assert synthetic.measured is False
    assert synthetic.label == "synthetic timing, not a measurement"
    assert synthetic.below_ceiling(1.0, 1.0)["measured"] is False
    assert synthetic.report_sha256 == SYNTHETIC["report_sha256"]


def test_a_tampered_report_sets_no_ceiling():
    report = fresh()
    report["summary"]["fma_ceiling_gflops_per_second"] *= 10

    with pytest.raises(GpuCeilingVerificationError, match="summary"):
        roofline_from_report(reseal(report))
    report = fresh()
    report["ceilings"]["bandwidth"]["gb_per_second"] = 1.0
    with pytest.raises(GpuCeilingVerificationError):
        roofline_from_report(report)


@pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("inf"), True, "8"])
def test_intensities_and_rates_must_be_positive_finite_numbers(value):
    line = roofline()

    with pytest.raises(GpuCeilingError, match="positive finite number"):
        line.attainable_gflops_per_second(value)
    with pytest.raises(GpuCeilingError, match="positive finite number"):
        line.below_ceiling(value, 8)
    with pytest.raises(GpuCeilingError, match="positive finite number"):
        roofline(compute=value)


def test_a_roofline_needs_a_digest_and_a_label_that_matches_its_kind():
    with pytest.raises(GpuCeilingError, match="SHA-256"):
        MeasuredRoofline(1.0, 1.0, "abc", True, "x")
    with pytest.raises(GpuCeilingError, match="label must match"):
        MeasuredRoofline(1.0, 1.0, DIGEST, True, "synthetic timing, not a measurement")


def test_a_report_file_loads_into_a_roofline(tmp_path):
    path = tmp_path / "recorded.json"
    path.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")

    assert load_roofline(path) == roofline_from_report(json.loads(RECORDED_REPORT_JSON))
