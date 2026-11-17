import copy
import json
from fractions import Fraction

import pytest

from ai_journey.gpu_ceiling_roofline import MeasuredRoofline
from ai_journey.kernel_metal_report import (
    ENVIRONMENT_FIELDS,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from ai_journey.matmul_metal_verify import verify_report as verify_matmul_report
from ai_journey.matmul_model_traffic import NO_REUSE_INTENSITY_LIMIT, account_matmul
from ai_journey.matmul_roofline_gap import (
    CEILING_SOURCE,
    DEFINITIONS,
    MATMUL_SOURCE,
    MEASUREMENT_BOUNDARY,
    SYNTHETIC_BOUNDARY,
    MatmulRooflineError,
    MatmulRooflineVerificationError,
    _point,
    build_gap_report,
    gap_from_files,
)
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import DIGEST_FIELD, report_digest
from ai_journey.memory_access_tiling import MatmulShape
from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as CEILING_JSON
from test_gpu_ceiling_verify import SYNTHETIC as SYNTHETIC_CEILINGS
from test_matmul_metal_recorded import RECORDED_REPORT_JSON as MATMUL_JSON
from test_matmul_metal_verify import synthetic_report

MATMUL = json.loads(MATMUL_JSON)
CEILINGS = json.loads(CEILING_JSON)
NAIVE = ("matmul_x_columns", "matmul_x_rows")
SIDES = (*NAIVE, "torch_matmul")


def resealed(report, change):
    """Change a copy of a sealed report, then seal it again."""

    tampered = copy.deepcopy(report)
    del tampered[DIGEST_FIELD]
    change(tampered)
    tampered[DIGEST_FIELD] = report_digest(tampered)
    return tampered


def with_environment(report, environment):
    # A synthetic report holds tokens only, so tests can give every source
    # the recorded host's tokens, and the pair then matches on any host.
    return resealed(report, lambda copied: copied["environment"].update(environment))


def synthetic_matmul():
    _, report = synthetic_report()
    return with_environment(report, MATMUL["environment"])


def synthetic_ceilings():
    return with_environment(SYNTHETIC_CEILINGS, CEILINGS["environment"])


def measured_gap():
    return build_gap_report(json.loads(MATMUL_JSON), json.loads(CEILING_JSON))


def write_sources(directory, matmul=MATMUL, ceilings=CEILINGS):
    """Write both sources as the canonical files that their loaders accept."""

    paths = directory / "matmul.json", directory / "ceilings.json"
    for path, report in zip(paths, (matmul, ceilings)):
        path.write_bytes(canonical_json_bytes(report) + b"\n")
    return paths


def test_two_measured_reports_give_a_measured_gap_on_any_host():
    gap = measured_gap()
    summary = CEILINGS["summary"]

    assert (gap["measured"], gap["label"], gap["measurement_scope"]) == (
        True,
        MEASUREMENT_LABEL,
        MEASUREMENT_SCOPE,
    )
    assert gap["evidence_boundary"] == MEASUREMENT_BOUNDARY
    assert gap["evidence_boundary"]["vendor_peak_used"] is False
    assert gap["sources"][MATMUL_SOURCE]["report_sha256"] == MATMUL[DIGEST_FIELD]
    assert gap["sources"][CEILING_SOURCE]["report_sha256"] == CEILINGS[DIGEST_FIELD]
    assert (
        gap["sources"][CEILING_SOURCE]["load_average_before"]
        == (summary["load_average_before"])
    )
    assert (
        gap["sources"][CEILING_SOURCE]["load_average_after"]
        == (summary["load_average_after"])
    )
    assert gap["environment"] == MATMUL["environment"]
    assert (
        gap["ceilings"]["fma_gflops_per_second"]
        == (summary["fma_ceiling_gflops_per_second"])
    )
    assert (
        gap["ceilings"]["bandwidth_gb_per_second"]
        == (summary["bandwidth_ceiling_gb_per_second"])
    )
    assert (
        gap["ceilings"]["ridge_flops_per_byte"]
        == (summary["ridge_point_flops_per_byte"])
    )
    # That small recorded run flags its own FMA ceiling as fixed-cost bound.
    assert gap["ceilings"]["fma_scales_with_work"] is False
    assert gap["definitions"] == DEFINITIONS
    assert gap[DIGEST_FIELD] == report_digest(gap)


def test_each_ratio_follows_from_the_sources_and_the_model_byte_counts():
    gap = measured_gap()
    fma = CEILINGS["summary"]["fma_ceiling_gflops_per_second"]
    bandwidth = CEILINGS["summary"]["bandwidth_ceiling_gb_per_second"]
    roofline = MeasuredRoofline(
        fma, bandwidth, CEILINGS[DIGEST_FIELD], True, MEASUREMENT_LABEL
    )

    # The share uses the measured FMA ceiling, not the best torch.matmul rate.
    assert fma != MATMUL["summary"]["measured_ceiling_gflops_per_second"]
    assert len(gap["points"]) == len(MATMUL["benchmark"]["points"]) == 2
    for point, source in zip(gap["points"], MATMUL["benchmark"]["points"]):
        m, n, k = source["m"], source["n"], source["k"]
        accounting = account_matmul(MatmulShape(m, n, k))
        assert (point["m"], point["n"], point["k"]) == (m, n, k)
        assert point["flops_per_call"] == 2 * m * n * k
        roofs = {}
        for model, traffic in (
            ("no_reuse", accounting.no_reuse),
            ("ideal_reuse", accounting.ideal_reuse),
        ):
            entry = point["intensities"][model]
            exact = Fraction(
                entry["flops_per_byte"]["numerator"],
                entry["flops_per_byte"]["denominator"],
            )
            assert exact == traffic.intensity
            assert entry["bytes"] == traffic.bytes
            roofs[model] = bandwidth * float(exact)
            assert entry["bandwidth_roof_gflops_per_second"] == roofs[model]
            assert entry["bound"] == roofline.bound(float(exact))
        assert accounting.no_reuse.intensity == Fraction(k, 4 * k + 2)
        assert accounting.no_reuse.intensity < NO_REUSE_INTENSITY_LIMIT
        for side in SIDES:
            entry = point["sides"][side]
            rate = source["sides"][side]["median_gflops_per_second"]
            median = source["sides"][side]["timing"]["median_ns"]
            assert (entry["median_gflops_per_second"], entry["median_sample_ns"]) == (
                rate,
                median,
            )
            assert entry["fma_ceiling_share"] == rate / fma
            assert entry["no_reuse_roof_share"] == rate / roofs["no_reuse"]
            assert entry["ideal_reuse_roof_share"] == rate / roofs["ideal_reuse"]
        for side, mapping in zip(NAIVE, ("x_columns", "x_rows")):
            assert point["sides"][side]["mapping"] == mapping
            assert (
                point["sides"][side]["torch_ratio"]
                == (source["naive_to_torch_gflops_ratio"][side])
            )
        assert "torch_ratio" not in point["sides"]["torch_matmul"]


def test_the_summary_names_the_best_share_and_each_range():
    gap = measured_gap()
    points = gap["points"]
    summary = gap["summary"]

    assert (summary["points"], summary["label"]) == (2, MEASUREMENT_LABEL)
    for side in SIDES:
        shares = [point["sides"][side]["fma_ceiling_share"] for point in points]
        assert summary["best_fma_ceiling_share"][side] == {
            "index": shares.index(max(shares)),
            "fma_ceiling_share": max(shares),
        }
    for side in NAIVE:
        for field, key in (
            ("no_reuse_roof_share", "no_reuse_roof_share_range"),
            ("ideal_reuse_roof_share", "ideal_reuse_roof_share_range"),
            ("torch_ratio", "torch_ratio_range"),
        ):
            values = [point["sides"][side][field] for point in points]
            assert summary[key][side] == [min(values), max(values)]
        above = [point["sides"][side]["no_reuse_roof_share"] > 1 for point in points]
        assert summary["points_above_no_reuse_roof"][side] == sum(above)


@pytest.mark.parametrize(
    "sources",
    [
        pytest.param(("measured", "synthetic"), id="synthetic-ceilings"),
        pytest.param(("synthetic", "measured"), id="synthetic-matmul"),
        pytest.param(("synthetic", "synthetic"), id="both-synthetic"),
    ],
)
def test_one_synthetic_source_makes_the_whole_gap_synthetic(sources):
    matmul = synthetic_matmul() if sources[0] == "synthetic" else MATMUL
    ceilings = synthetic_ceilings() if sources[1] == "synthetic" else CEILINGS

    gap = build_gap_report(matmul, ceilings)

    assert (gap["measured"], gap["label"], gap["measurement_scope"]) == (
        False,
        SYNTHETIC_LABEL,
        SYNTHETIC_SCOPE,
    )
    assert gap["evidence_boundary"] == SYNTHETIC_BOUNDARY
    assert gap["summary"]["label"] == SYNTHETIC_LABEL
    assert [
        gap["sources"][MATMUL_SOURCE]["measured"],
        gap["sources"][CEILING_SOURCE]["measured"],
    ] == [source == "measured" for source in sources]


@pytest.mark.parametrize("field", ENVIRONMENT_FIELDS)
def test_reports_that_name_other_software_or_platforms_do_not_pair(field):
    ceilings = with_environment(synthetic_ceilings(), {field: "other-1.0"})

    with pytest.raises(MatmulRooflineError, match=f"different software.*{field}"):
        build_gap_report(MATMUL, ceilings)


def test_a_tampered_or_swapped_source_sets_no_number():
    tampered = resealed(
        MATMUL,
        lambda report: report["summary"].update(measured_ceiling_gflops_per_second=1.0),
    )

    with pytest.raises(ValueError, match="summary"):
        verify_matmul_report(tampered)
    with pytest.raises(MatmulRooflineVerificationError, match="^matmul-metal report"):
        build_gap_report(tampered, CEILINGS)
    with pytest.raises(MatmulRooflineVerificationError, match="^gpu-ceilings report"):
        build_gap_report(MATMUL, MATMUL)
    with pytest.raises(MatmulRooflineVerificationError, match="^matmul-metal report"):
        build_gap_report(CEILINGS, CEILINGS)


def test_files_load_through_the_verified_canonical_loaders(tmp_path):
    matmul, ceilings = write_sources(tmp_path)

    assert gap_from_files(matmul, ceilings) == measured_gap()
    with pytest.raises(MatmulRooflineError, match="^gpu-ceilings report") as missing:
        gap_from_files(matmul, tmp_path / "missing.json")
    assert not isinstance(missing.value, MatmulRooflineVerificationError)
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(MATMUL, indent=1) + "\n", encoding="ascii")
    with pytest.raises(MatmulRooflineVerificationError, match="not canonical"):
        gap_from_files(reformatted, ceilings)


def test_a_point_whose_sides_ran_other_batches_has_no_torch_ratio():
    # The torch ratio is a ratio of sample times, which is a ratio of rates
    # only when every side ran the same calls in each sample.
    point = copy.deepcopy(MATMUL["benchmark"]["points"][0])
    point["sides"]["torch_matmul"]["calls_per_sample"] += 1
    summary = CEILINGS["summary"]
    roofline = MeasuredRoofline(
        summary["fma_ceiling_gflops_per_second"],
        summary["bandwidth_ceiling_gb_per_second"],
        CEILINGS[DIGEST_FIELD],
        True,
        MEASUREMENT_LABEL,
    )

    with pytest.raises(MatmulRooflineError, match="different calls per sample"):
        _point(point, roofline)
