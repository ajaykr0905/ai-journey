import copy
import json
from pathlib import Path

import pytest

from ai_journey.matmul_bottleneck_config import (
    MatmulBottleneckError,
    MatmulBottleneckVerificationError,
    load_config,
)
from ai_journey.matmul_bottleneck_sources import (
    join_points,
    sources_from_files,
    sources_from_reports,
)
from ai_journey.matmul_bottleneck_timing import timing_from_mapping
from ai_journey.matmul_roofline_gap import build_gap_report
from ai_journey.memory_access_plan import canonical_json_bytes
from test_matmul_bottleneck_timing import SYNTHETIC_TIMES, synthetic_timing_mapping
from test_matmul_roofline_gap import (
    CEILING_JSON,
    MATMUL_JSON,
    resealed,
    synthetic_ceilings,
    synthetic_matmul,
)

ROOT = Path(__file__).resolve().parents[1]
RULE = load_config(ROOT / "config/day-75-matmul-bottleneck.json").rule


def recorded_reports():
    matmul, ceilings = json.loads(MATMUL_JSON), json.loads(CEILING_JSON)
    return matmul, ceilings, build_gap_report(matmul, ceilings)


def recorded_sources():
    return sources_from_reports(*recorded_reports())


def synthetic_timing(times=SYNTHETIC_TIMES):
    return timing_from_mapping(synthetic_timing_mapping(times))


def write_reports(directory, reports):
    paths = tuple(
        directory / name for name in ("matmul.json", "ceilings.json", "gap.json")
    )
    for path, report in zip(paths, reports):
        path.write_bytes(canonical_json_bytes(report) + b"\n")
    return paths


def test_each_naive_kernel_at_each_gap_point_joins_its_timing():
    sources = recorded_sources()
    joined = join_points(sources, synthetic_timing(), RULE)
    gap = sources.gap_report

    assert sources.group_size == (32, 8)
    assert sources.simd_widths == {"matmul_x_columns": 32, "matmul_x_rows": 32}
    assert [(item.index, item.kernel) for item in joined] == [
        (0, "matmul_x_columns"),
        (0, "matmul_x_rows"),
        (1, "matmul_x_columns"),
        (1, "matmul_x_rows"),
    ]
    for item in joined:
        point = gap["points"][item.index]
        side = point["sides"][item.kernel]
        dims = (item.shape.m, item.shape.n, item.shape.k)
        gpu, host, dispatch = SYNTHETIC_TIMES[dims][item.kernel]
        assert item.inputs.steady_ns_per_call == (
            side["median_sample_ns"] / point["calls_per_sample"]
        )
        assert item.inputs.rate_gflops_per_second == side["median_gflops_per_second"]
        assert (item.inputs.gpu_ns, item.inputs.host_ns, item.inputs.dispatch_ns) == (
            gpu,
            host,
            dispatch,
        )
        assert (
            item.inputs.bandwidth_roof_gflops_per_second
            == (point["intensities"]["ideal_reuse"]["bandwidth_roof_gflops_per_second"])
        )
        assert (
            item.inputs.fma_ceiling_gflops_per_second
            == (gap["ceilings"]["fma_gflops_per_second"])
        )
        # The small recorded ceiling run flags its FMA ceiling as fixed-cost bound.
        assert item.inputs.fma_scales_with_work is False


def test_files_load_through_the_day_74_loaders(tmp_path):
    paths = write_reports(tmp_path, recorded_reports())
    sources = sources_from_files(*paths)

    assert sources.gap_summary["sources_checked"] is True
    assert sources.gap_report == recorded_reports()[2]


def test_a_resealed_gap_with_another_median_is_rejected():
    matmul, ceilings, gap = recorded_reports()

    def faster(report):
        report["points"][0]["sides"]["matmul_x_rows"]["median_sample_ns"] -= 1

    with pytest.raises(MatmulBottleneckVerificationError, match="roofline gap report"):
        sources_from_reports(matmul, ceilings, resealed(gap, faster))


def test_a_resealed_source_fails_its_own_verifier_first():
    matmul, ceilings, gap = recorded_reports()
    tampered = resealed(matmul, lambda report: report["summary"].update(points=3))

    with pytest.raises(MatmulBottleneckVerificationError, match="matmul-metal report"):
        sources_from_reports(tampered, ceilings, gap)


def test_a_gap_from_other_sources_does_not_pair():
    matmul, ceilings, _ = recorded_reports()
    other = build_gap_report(synthetic_matmul(), synthetic_ceilings())

    with pytest.raises(MatmulBottleneckVerificationError, match="supplied source"):
        sources_from_reports(matmul, ceilings, other)


def test_the_timing_must_cover_every_naive_kernel_and_name_the_same_software():
    sources = recorded_sources()
    missing = copy.deepcopy(SYNTHETIC_TIMES)
    del missing[(129, 65, 1000)]["matmul_x_rows"]
    with pytest.raises(MatmulBottleneckError, match="no matmul_x_rows entry at 129"):
        join_points(sources, synthetic_timing(missing), RULE)

    mapping = synthetic_timing_mapping()
    mapping["environment"]["torch_version"] = "2.14.0"
    with pytest.raises(MatmulBottleneckError, match="other software.*torch_version"):
        join_points(sources, timing_from_mapping(mapping), RULE)


def test_extra_timing_entries_are_ignored():
    extra = copy.deepcopy(SYNTHETIC_TIMES)
    extra[(128, 128, 128)] = {"matmul_x_columns": (1, 2, 1)}

    assert len(join_points(recorded_sources(), synthetic_timing(extra), RULE)) == 4


def test_unreadable_files_are_invalid_input(tmp_path):
    paths = write_reports(tmp_path, recorded_reports())
    paths[2].write_text("{", encoding="utf-8")

    with pytest.raises(MatmulBottleneckError, match="roofline gap report") as caught:
        sources_from_files(*paths)
    assert not isinstance(caught.value, MatmulBottleneckVerificationError)
