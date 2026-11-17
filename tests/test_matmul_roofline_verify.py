import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.gpu_ceiling_roofline import MeasuredRoofline
from ai_journey.kernel_metal_report import MEASUREMENT_LABEL, SYNTHETIC_LABEL
from ai_journey.matmul_roofline_gap import (
    CEILING_SOURCE,
    MATMUL_SOURCE,
    MEASUREMENT_BOUNDARY,
    MatmulRooflineVerificationError,
    _point,
    _summary,
    build_gap_report,
)
from ai_journey.matmul_roofline_verify import verify_gap_report
from ai_journey.memory_access_report import DIGEST_FIELD
from test_matmul_roofline_gap import (
    CEILINGS,
    MATMUL,
    measured_gap,
    resealed,
    synthetic_ceilings,
    synthetic_matmul,
)

ROOT = Path(__file__).resolve().parents[1]
GAP = measured_gap()


def setter(*path, value):
    def change(report):
        target = report
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value(target[path[-1]]) if callable(value) else value

    return change


def test_a_measured_gap_verifies_and_says_its_sources_were_not_rebuilt():
    summary = verify_gap_report(GAP)

    assert summary == {
        "kind": "matmul_roofline_gap",
        "report_sha256": GAP[DIGEST_FIELD],
        "measured": True,
        "scope": "apple_gpu_metal_this_host",
        "label": MEASUREMENT_LABEL,
        "points": 2,
        "matmul_report_sha256": MATMUL[DIGEST_FIELD],
        "ceiling_report_sha256": CEILINGS[DIGEST_FIELD],
        "sources_checked": False,
    }


def test_a_gap_rebuilt_from_its_sources_binds_it_to_them():
    rebuilt = build_gap_report(MATMUL, CEILINGS)

    assert verify_gap_report(GAP, rebuilt=rebuilt)["sources_checked"] is True
    other = build_gap_report(MATMUL, synthetic_ceilings())
    with pytest.raises(MatmulRooflineVerificationError, match="supplied source"):
        verify_gap_report(GAP, rebuilt=other)


def test_a_synthetic_gap_verifies_as_synthetic():
    gap = build_gap_report(synthetic_matmul(), synthetic_ceilings())

    summary = verify_gap_report(gap)

    assert (summary["measured"], summary["scope"], summary["label"]) == (
        False,
        "synthetic_backend_or_clock",
        SYNTHETIC_LABEL,
    )


def test_consistent_new_ceilings_pass_alone_and_fail_against_the_sources():
    # Doubling the FMA ceiling and recalculating every share makes a report
    # that adds up. Only a rebuild from the verified sources can catch it.
    fma = 2 * CEILINGS["summary"]["fma_ceiling_gflops_per_second"]
    bandwidth = CEILINGS["summary"]["bandwidth_ceiling_gb_per_second"]
    roofline = MeasuredRoofline(
        fma, bandwidth, CEILINGS[DIGEST_FIELD], True, MEASUREMENT_LABEL
    )

    def doubled(report):
        report["ceilings"]["fma_gflops_per_second"] = fma
        report["ceilings"]["ridge_flops_per_byte"] = fma / bandwidth
        report["points"] = [
            _point(point, roofline) for point in MATMUL["benchmark"]["points"]
        ]
        report["summary"] = _summary(MEASUREMENT_LABEL, report["points"])

    tampered = resealed(GAP, doubled)

    assert verify_gap_report(tampered)["sources_checked"] is False
    with pytest.raises(MatmulRooflineVerificationError, match="rebuilt from its"):
        verify_gap_report(tampered, rebuilt=build_gap_report(MATMUL, CEILINGS))


def test_a_changed_byte_without_a_new_seal_fails_on_the_digest():
    report = json.loads(json.dumps(GAP))
    report["points"][0]["sides"]["matmul_x_columns"]["torch_ratio"] = 9.0

    with pytest.raises(MatmulRooflineVerificationError, match="digest"):
        verify_gap_report(report)


TAMPERS = [
    ("schema", setter("schema_version", value=2), "schema"),
    ("kind", setter("kind", value="other"), "kind"),
    ("extra-field", setter("note", value="x"), "fields do not match"),
    ("measured-flag", setter("measured", value=False), "only when both sources"),
    ("label", setter("label", value=SYNTHETIC_LABEL), "labels or boundary"),
    (
        "boundary-flag",
        setter("evidence_boundary", "vendor_peak_used", value=True),
        "labels or boundary",
    ),
    (
        "boundary-text",
        setter("evidence_boundary", "statement", value="A peak."),
        "labels or boundary",
    ),
    (
        "source-kind",
        setter("sources", CEILING_SOURCE, "kind", value="other"),
        "another kind",
    ),
    (
        "source-digest",
        setter("sources", MATMUL_SOURCE, "report_sha256", value="abc"),
        "SHA-256",
    ),
    (
        "source-measured",
        setter("sources", MATMUL_SOURCE, "measured", value=False),
        "label does not match",
    ),
    (
        "source-label",
        setter("sources", CEILING_SOURCE, "label", value=SYNTHETIC_LABEL),
        "label does not match",
    ),
    (
        "load-average",
        setter("sources", CEILING_SOURCE, "load_average_before", value=[1.0, 2.0]),
        "three nonnegative floats",
    ),
    (
        "load-average-negative",
        setter("sources", CEILING_SOURCE, "load_average_after", value=[1.0, -2.0, 3.0]),
        "three nonnegative floats",
    ),
    (
        "source-field",
        setter("sources", MATMUL_SOURCE, "load_average_before", value=[1.0, 1.0, 1.0]),
        "fields do not match",
    ),
    ("environment-field", setter("environment", "host", value="x"), "environment"),
    (
        "environment-token",
        setter("environment", "torch_version", value="2.15 beta"),
        "short tokens",
    ),
    (
        "environment-platform",
        setter("environment", "platform_system", value="Linux"),
        "macOS",
    ),
    (
        "fma-ceiling",
        setter("ceilings", "fma_gflops_per_second", value=lambda value: value * 2),
        "ceilings section",
    ),
    (
        "integer-ceiling",
        setter("ceilings", "bandwidth_gb_per_second", value=82),
        "positive finite float",
    ),
    (
        "scales-flag",
        setter("ceilings", "fma_scales_with_work", value=0),
        "ceilings section",
    ),
    ("ceiling-text", setter("ceilings", "definition", value="peak"), "ceilings"),
    ("definitions", setter("definitions", "bound", value="x"), "definitions"),
    ("no-points", setter("points", value=[]), "1 to 16 points"),
    (
        "reordered-points",
        setter("points", value=lambda points: points[::-1]),
        "point 0 does not recompute",
    ),
    (
        "index",
        setter("points", 0, "index", value=1),
        "point 0 does not recompute",
    ),
    ("shape", setter("points", 0, "m", value=0), "invalid shape or batch"),
    ("batch", setter("points", 0, "calls_per_sample", value=4097), "invalid shape"),
    (
        "median-type",
        setter("points", 0, "sides", "torch_matmul", "median_sample_ns", value=1.0),
        "positive integers",
    ),
    (
        "side-missing",
        setter("points", 1, "sides", value=lambda sides: {"torch_matmul": {}}),
        "three sides",
    ),
    (
        "rate",
        setter(
            "points",
            0,
            "sides",
            "matmul_x_rows",
            "median_gflops_per_second",
            value=lambda value: value * 1.5,
        ),
        "point 0 does not recompute",
    ),
    (
        "fma-share",
        setter(
            "points",
            1,
            "sides",
            "matmul_x_columns",
            "fma_ceiling_share",
            value=lambda value: value / 2,
        ),
        "point 1 does not recompute",
    ),
    (
        "roof-share",
        setter(
            "points",
            0,
            "sides",
            "torch_matmul",
            "no_reuse_roof_share",
            value=lambda value: value + 1,
        ),
        "point 0 does not recompute",
    ),
    (
        "torch-ratio",
        setter(
            "points",
            0,
            "sides",
            "matmul_x_columns",
            "torch_ratio",
            value=lambda value: 1 / value,
        ),
        "point 0 does not recompute",
    ),
    (
        "mapping",
        setter("points", 0, "sides", "matmul_x_rows", "mapping", value="x_columns"),
        "point 0 does not recompute",
    ),
    (
        "bound",
        setter("points", 1, "intensities", "ideal_reuse", "bound", value="compute"),
        "point 1 does not recompute",
    ),
    (
        "intensity",
        setter(
            "points",
            0,
            "intensities",
            "no_reuse",
            "flops_per_byte",
            "decimal",
            value="0.25",
        ),
        "point 0 does not recompute",
    ),
    (
        "roof",
        setter(
            "points",
            1,
            "intensities",
            "no_reuse",
            "bandwidth_roof_gflops_per_second",
            value=lambda value: value * 2,
        ),
        "point 1 does not recompute",
    ),
    ("summary-count", setter("summary", "points", value=3), "summary"),
    (
        "summary-best",
        setter(
            "summary",
            "best_fma_ceiling_share",
            "torch_matmul",
            "index",
            value=lambda index: 1 - index,
        ),
        "summary",
    ),
    (
        "summary-above",
        setter(
            "summary",
            "points_above_no_reuse_roof",
            "matmul_x_rows",
            value=lambda count: count + 1,
        ),
        "summary",
    ),
]


@pytest.mark.parametrize(
    ("change", "message"),
    [pytest.param(change, message, id=name) for name, change, message in TAMPERS],
)
def test_each_resealed_tamper_fails_at_its_own_check(change, message):
    tampered = resealed(GAP, change)

    with pytest.raises(MatmulRooflineVerificationError, match=message):
        verify_gap_report(tampered)


def test_the_tamper_values_are_changes():
    for name, change, _ in TAMPERS:
        assert resealed(GAP, change) != GAP, name


def test_structure_errors_are_verification_errors():
    for report in (None, [], {"kind": "matmul_roofline_gap"}, {**GAP, "points": None}):
        with pytest.raises(MatmulRooflineVerificationError):
            verify_gap_report(report)
    broken = resealed(GAP, setter("points", 0, value=None))
    with pytest.raises(MatmulRooflineVerificationError, match="point 0"):
        verify_gap_report(broken)


def test_the_measurement_boundary_rules_out_peaks_cuda_and_traffic_claims():
    boundary = GAP["evidence_boundary"]

    assert boundary == MEASUREMENT_BOUNDARY
    for flag in (
        "vendor_peak_used",
        "traffic_measured",
        "cuda_kernel_executed",
        "nvidia_gpu_used",
        "distributed_claim",
        "production_claim",
        "portable_performance_claim",
    ):
        assert boundary[flag] is False, flag
    assert boundary["host_specific"] is True
    statement = boundary["statement"]
    assert "host-specific Apple GPU (Metal) measurements" in statement
    assert "not a vendor peak and not a hardware limit" in statement
    assert "not from measured traffic" in statement


def test_the_verifier_runs_where_torch_cannot_be_imported(tmp_path):
    path = tmp_path / "gap.json"
    path.write_text(json.dumps(GAP), encoding="ascii")
    script = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.matmul_roofline_verify import verify_gap_report\n"
        "summary = verify_gap_report(json.loads(open(sys.argv[1]).read()))\n"
        "print(summary['report_sha256'], 'torch' in sys.modules and "
        "sys.modules['torch'] is not None)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"{GAP[DIGEST_FIELD]} False\n"
