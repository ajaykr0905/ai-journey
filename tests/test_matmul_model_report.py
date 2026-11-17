from dataclasses import replace
from hashlib import sha256

import pytest

from ai_journey import matmul_model_report
from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.matmul_model_cuda import CHECK_NAMES, cuda_source
from ai_journey.matmul_model_plan import (
    MAX_CASE_INNER,
    MAX_CASE_MULTIPLY_ADDS,
    MAX_CASES,
    MAX_CEILINGS,
    MatmulModelPlan,
)
from ai_journey.matmul_model_reference import MatmulModelError
from ai_journey.matmul_model_report import (
    DIGEST_FIELD,
    EVIDENCE_BOUNDARY,
    REPORT_KIND,
    build_report,
    report_digest,
    seal_report,
)
from ai_journey.matmul_model_verify import MAX_REPORT_BYTES
from test_matmul_model_plan import plan_mapping

PLAN = MatmulModelPlan.from_mapping(plan_mapping())


@pytest.fixture(scope="module")
def report():
    return build_report(PLAN)


def float_free(value):
    if type(value) is float:
        return False
    if isinstance(value, dict):
        return all(float_free(item) for item in value.values())
    if isinstance(value, list):
        return all(float_free(item) for item in value)
    return True


def test_report_holds_every_section_and_a_fixed_cpu_only_boundary(report):
    assert set(report) == {
        "schema_version",
        "kind",
        "plan",
        "limits",
        "rule",
        "geometry",
        "cases",
        "accounting",
        "occupancy",
        "cuda_source",
        "summary",
        "evidence_boundary",
    }
    assert (report["schema_version"], report["kind"]) == (1, REPORT_KIND)
    assert report["plan"] == plan_mapping()
    assert report["evidence_boundary"] == dict(EVIDENCE_BOUNDARY)
    for flag in (
        "gpu_executed",
        "cuda_compiled",
        "timing_measured",
        "ceilings_measured",
    ):
        assert report["evidence_boundary"][flag] is False
    assert report["limits"]["max_block_dim"] == [1024, 1024, 64]
    assert report["rule"]["input_streams"] == ["matmul-a", "matmul-b"]
    assert report["geometry"]["cache_modeled"] is False
    assert float_free(report)


def test_every_run_matches_the_reference_in_every_order(report):
    ragged, aligned = report["cases"]

    assert [section["mapping"] for section in ragged["mappings"]] == [
        "x_columns",
        "x_rows",
    ]
    # 3 x 35 elements in 32 x 2 blocks: x over columns launches 2 x 2 blocks.
    columns = ragged["mappings"][0]
    assert columns["launch"]["grid"] == [2, 2, 1]
    assert (columns["launch"]["useful_threads"], columns["launch"]["idle_threads"]) == (
        105,
        151,
    )
    assert (columns["ragged"], aligned["mappings"][0]["ragged"]) == (True, False)
    for case in (ragged, aligned):
        for section in case["mappings"]:
            assert section["order_independent"] is True
            assert [run["order"]["kind"] for run in section["runs"]] == [
                "canonical",
                "permuted",
            ]
            for run in section["runs"]:
                assert run["output_sha256"] == case["reference_sha256"]
                assert run["matches_reference"] is True
                assert (
                    run["reads"]
                    == 2 * case["shape"]["m"] * case["shape"]["n"] * case["shape"]["k"]
                )
                assert run["out_of_bounds_writes"] == run["write_conflicts"] == 0


def test_access_sections_class_each_mapping_and_trace_two_probe_warps(report):
    columns, rows = report["cases"][0]["mappings"]

    def classes(section, name):
        return section["access"]["operands"][name]["classes"]

    # Every A load is a broadcast and every B load coalesces when x runs over
    # columns; x over rows strides A and broadcasts B.
    for section, a_class, b_class in (
        (columns, "broadcast", "coalesced"),
        (rows, "strided", "broadcast"),
    ):
        a, b = classes(section, "a"), classes(section, "b")
        assert a[a_class] == sum(a.values()) > 0
        assert b[b_class] == sum(b.values()) > 0
    for section in (columns, rows):
        assert section["cross_check"]["passed"] is True
        assert [probe["element"] for probe in section["probes"]] == [[0, 0], [2, 34]]
        for probe in section["probes"]:
            assert probe["trace_matches_model"] is True
            assert [step["step"] for step in probe["steps"]] == [0, 1, 2]
    # Row 2 of columns 32..63 keeps 3 lanes; rows 0..31 of column 34 keep 3.
    assert [probe["active_lanes"] for probe in columns["probes"]] == [32, 3]
    assert [probe["active_lanes"] for probe in rows["probes"]] == [3, 3]
    first = columns["probes"][0]["steps"][1]
    assert first["a"]["lane_addresses"] == [4] * 32
    assert first["b"]["lane_addresses"] == [4 * (35 + lane) for lane in range(32)]


def test_accounting_rooflines_occupancy_and_cuda_cover_every_declared_input(report):
    assert [item["source"] for item in report["accounting"]] == [
        "case:ragged",
        "case:aligned",
        "declared",
    ]
    for item in report["accounting"]:
        assert item["cross_checks"] == {
            "memory_access_naive_traffic": True,
            "roofline_matmul_workload": True,
        }
        (point,) = item["roofline"]
        assert point["ceilings"]["label"] == "declared-ridge-4"
        assert point["no_reuse"]["bound"] == "memory"
    declared = report["accounting"][2]["roofline"][0]["ideal_reuse"]
    assert (declared["bound"], declared["attainable_gflops"]["decimal"]) == (
        "compute",
        "4.000000",
    )
    occupancy = report["occupancy"]
    assert occupancy["threads_per_block"] == 64
    assert [
        item["kernel"]["registers_per_thread"] for item in occupancy["estimates"]
    ] == [32, 64]
    cuda = report["cuda_source"]
    assert cuda["text"] == cuda_source(block=(32, 2))
    assert cuda["sha256"] == sha256(cuda["text"].encode("ascii")).hexdigest()
    assert (cuda["compiled"], cuda["executed"]) == (False, False)
    for item in cuda["structure_by_case"]:
        assert item["passed"] is True
        assert list(item["checks"]) == list(CHECK_NAMES)
    assert report["summary"] == {
        "cases": 2,
        "launches": 4,
        "ragged_launches": 2,
        "runs": 8,
        "runs_matching_reference": 8,
        "memory_safe_runs": 8,
        "race_free_runs": 8,
        "order_independent_launches": 4,
        "cross_checked_launches": 4,
        "cross_checked_requests": report["summary"]["cross_checked_requests"],
        "probes": 8,
        "probe_traces_matching_model": 8,
        "accounting_shapes": 3,
        "roofline_points": 6,
        "memory_bound_points": 5,
        "occupancy_estimates": 2,
        "cuda_checks_passed": 16,
        "cuda_checks": 16,
    }


def test_a_broken_simulator_changes_the_runs_it_reports(monkeypatch):
    real = matmul_model_report.run_naive_matmul

    def drifted(a, b, shape, **options):
        run = real(a, b, shape, **options)
        if options["order"].kind == "permuted":
            return replace(run, output_sha256="0" * 64)
        return run

    monkeypatch.setattr(matmul_model_report, "run_naive_matmul", drifted)
    section = build_report(PLAN)["cases"][0]["mappings"][0]

    assert [run["matches_reference"] for run in section["runs"]] == [True, False]
    assert section["order_independent"] is False


def test_sealing_adds_a_digest_over_every_other_field(report):
    sealed = seal_report(report)

    assert sealed[DIGEST_FIELD] == report_digest(report) == report_digest(sealed)
    assert sealed[DIGEST_FIELD] == sha256(canonical_json_bytes(report)).hexdigest()
    edited = {**report, "summary": {**report["summary"], "runs": 9}}
    assert report_digest(edited) != sealed[DIGEST_FIELD]
    with pytest.raises(MatmulModelError, match="already sealed"):
        seal_report(sealed)
    with pytest.raises(TypeError, match="MatmulModelPlan"):
        build_report(plan_mapping())


def test_the_largest_valid_case_keeps_four_copies_under_the_read_limit():
    # The probes list every loop step, so k and the address widths set most
    # of the report size. 512 x 4 x 64 has the most multiply-adds and steps,
    # and its x-over-rows probes hold 32 six-digit A addresses per step.
    mapping = {
        **plan_mapping(),
        "block": [32, 8],
        "cases": [
            {"name": "widest", "m": 512, "n": 4, "k": MAX_CASE_INNER, "input_seed": 1}
        ],
        "accounting_shapes": [
            {"m": 65536, "n": 65536, "k": k} for k in (1, 2, 3, 65536)
        ],
        "registers_per_thread": [16, 32, 40, 64, 96, 128],
        "ceilings": [
            {
                "label": f"pair-{index}",
                "compute_gflops": "123456789012.12345",
                "bandwidth_gbps": "0.00001",
            }
            for index in range(MAX_CEILINGS)
        ],
    }
    report = seal_report(build_report(MatmulModelPlan.from_mapping(mapping)))
    case = len(canonical_json_bytes(report["cases"]))
    rest = len(canonical_json_bytes(report)) + 1 - case

    assert 512 * 4 * MAX_CASE_INNER == MAX_CASE_MULTIPLY_ADDS
    # Three more cases also add accounting rows and order rows: under 40 KB.
    assert MAX_CASES * case + rest < 0.7 * MAX_REPORT_BYTES
