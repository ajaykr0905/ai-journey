from dataclasses import replace
from hashlib import sha256

import pytest

from ai_journey import kernel_launch_report
from ai_journey.kernel_launch_config import KernelLaunchError
from ai_journey.kernel_launch_cuda import cuda_source
from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.kernel_launch_plan import KernelLaunchPlan
from ai_journey.kernel_launch_report import (
    DIGEST_FIELD,
    EVIDENCE_BOUNDARY,
    REPORT_KIND,
    build_report,
    report_digest,
    seal_report,
)
from test_kernel_launch_plan import plan_mapping

PLAN = KernelLaunchPlan.from_mapping(plan_mapping())


@pytest.fixture(scope="module")
def report():
    return build_report(PLAN)


def test_report_holds_every_section_and_a_fixed_cpu_only_boundary(report):
    assert set(report) == {
        "schema_version",
        "kind",
        "plan",
        "limits",
        "grayscale_rule",
        "vector_add",
        "rgb_to_grayscale",
        "unguarded_demos",
        "cuda_source",
        "summary",
        "evidence_boundary",
    }
    assert (report["schema_version"], report["kind"]) == (1, REPORT_KIND)
    assert report["plan"] == plan_mapping()
    assert report["evidence_boundary"] == dict(EVIDENCE_BOUNDARY)
    assert report["evidence_boundary"]["gpu_executed"] is False
    assert report["evidence_boundary"]["cuda_compiled"] is False
    assert report["limits"]["max_block_dim"] == [1024, 1024, 64]
    assert report["grayscale_rule"]["weights_per_10000"] == [2989, 5870, 1140]


def test_guarded_runs_count_useful_and_idle_threads_and_agree(report):
    vector = report["vector_add"]
    gray = report["rgb_to_grayscale"]

    # 100 elements in blocks of 32 launch 128 threads; 10 x 6 pixels in 4 x 4
    # blocks launch a 3 x 2 grid of 12 x 8 threads.
    assert vector["launch"]["grid"] == [4, 1, 1]
    assert (vector["launch"]["useful_threads"], vector["launch"]["idle_threads"]) == (
        100,
        28,
    )
    assert gray["launch"]["grid"] == [3, 2, 1]
    assert (gray["launch"]["useful_threads"], gray["launch"]["idle_threads"]) == (
        60,
        36,
    )
    for section in (vector, gray):
        assert [run["order"] for run in section["runs"]] == [
            {"kind": "canonical", "seed": None},
            {"kind": "permuted", "seed": 5},
        ]
        assert section["order_independent"] is True
        assert len({run["order_sha256"] for run in section["runs"]}) == 2
        for run in section["runs"]:
            assert run["output_sha256"] == section["reference_sha256"]
            assert run["matches_reference"] is True
            assert run["useful_threads"] == section["launch"]["useful_threads"]
            assert run["threads_run"] == section["launch"]["launched_threads"]
            assert run["out_of_bounds_writes"] == run["write_conflicts"] == 0
    assert vector["runs"][0]["reads"] == 200
    assert gray["runs"][0]["reads"] == 180


def test_unguarded_demos_catch_ragged_writes_and_spare_aligned_ones(report):
    ragged_vector, ragged_image, aligned_image = report["unguarded_demos"]

    assert ragged_vector["kernel"] == "vector_add_unguarded"
    assert ragged_vector["ragged"] is True
    assert ragged_vector["out_of_bounds_writes"] == 28
    assert ragged_vector["first_out_of_bounds_writes"][0] == {
        "buffer": "out",
        "access": "write",
        "index": 100,
        "block": [3, 0, 0],
        "thread": [4, 0, 0],
    }
    assert len(ragged_vector["first_out_of_bounds_writes"]) == 4
    assert ragged_image["kernel"] == "rgb_to_grayscale_unguarded"
    assert ragged_image["out_of_bounds_writes"] > 0
    assert ragged_image["write_conflicts"] == ragged_image["extra_writes"] > 0
    assert aligned_image["ragged"] is False
    assert aligned_image["out_of_bounds_writes"] == 0
    assert aligned_image["write_conflicts"] == 0
    assert aligned_image["first_out_of_bounds_writes"] == []
    for demo in report["unguarded_demos"]:
        assert demo["in_bounds_output_matches_reference"] is True
        assert demo["orders_agree"] is True
        assert demo["orders_checked"] == 2


def test_cuda_section_pins_the_uncompiled_text_and_its_checks(report):
    cuda = report["cuda_source"]
    text = cuda_source(threads_per_block=32, block=(4, 4))

    assert cuda["text"] == text
    assert cuda["sha256"] == sha256(text.encode("ascii")).hexdigest()
    assert (cuda["compiled"], cuda["executed"]) == (False, False)
    assert cuda["structure"]["passed"] is True
    assert cuda["structure"]["launches"]["rgb_to_grayscale"] == {
        "grid": [3, 2, 1],
        "block": [4, 4, 1],
    }


def test_summary_totals_the_sections(report):
    assert report["summary"] == {
        "kernels": 2,
        "runs": 4,
        "runs_matching_reference": 4,
        "race_free_runs": 4,
        "order_independent_kernels": 2,
        "unguarded_demos": 3,
        "ragged_demos_with_out_of_bounds_writes": 2,
        "ragged_demos": 2,
        "aligned_demos_without_misuse": 1,
        "aligned_demos": 1,
        "out_of_bounds_writes_caught": 28
        + report["unguarded_demos"][1]["out_of_bounds_writes"],
        "write_conflicts_caught": report["unguarded_demos"][1]["write_conflicts"],
        "cuda_checks_passed": 10,
        "cuda_checks": 10,
    }


def test_builds_are_deterministic_and_seals_cover_every_other_field(report):
    sealed = seal_report(report)

    assert canonical_json_bytes(build_report(PLAN)) == canonical_json_bytes(report)
    assert DIGEST_FIELD not in report
    assert sealed[DIGEST_FIELD] == sha256(canonical_json_bytes(report)).hexdigest()
    assert report_digest(sealed) == sealed[DIGEST_FIELD]
    with pytest.raises(KernelLaunchError, match="already sealed"):
        seal_report(sealed)
    with pytest.raises(TypeError, match="KernelLaunchPlan"):
        build_report(plan_mapping())


def test_report_records_a_broken_run_instead_of_claiming_success(monkeypatch):
    real = kernel_launch_report.run_vector_add

    def broken(a, b, **options):
        run = real(a, b, **options)
        if options["order"].kind == "permuted":
            return replace(run, output_sha256="0" * 64)
        return run

    monkeypatch.setattr(kernel_launch_report, "run_vector_add", broken)
    report = build_report(PLAN)

    assert report["vector_add"]["order_independent"] is False
    assert [run["matches_reference"] for run in report["vector_add"]["runs"]] == [
        True,
        False,
    ]
    assert report["summary"]["runs_matching_reference"] == 3
    assert report["summary"]["order_independent_kernels"] == 1
