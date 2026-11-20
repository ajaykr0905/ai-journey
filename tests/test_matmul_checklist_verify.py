import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.matmul_checklist_config import config_from_mapping
from ai_journey.matmul_checklist_verify import (
    MatmulChecklistVerificationError,
    verify_report,
)
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import report_digest
from test_matmul_checklist_config import make_config_mapping
from test_matmul_checklist_correctness import StandInBackend
from test_matmul_checklist_report import stand_in_report

ROOT = Path(__file__).resolve().parents[1]
CONFIG = config_from_mapping(make_config_mapping())


def loaded(report):
    """Round-trip through canonical JSON, as a report file does."""

    return json.loads(canonical_json_bytes(report))


def resealed(report, mutate):
    changed = copy.deepcopy(report)
    mutate(changed)
    changed.pop("report_sha256")
    return {**changed, "report_sha256": report_digest(changed)}


@pytest.fixture(scope="module")
def report():
    return loaded(stand_in_report())


@pytest.fixture(scope="module")
def drifted():
    return loaded(stand_in_report(StandInBackend(drift=("matmul_s5_pad4",))))


@pytest.fixture(scope="module")
def refused():
    return loaded(
        stand_in_report(StandInBackend(threads={"matmul_s6_group_32x32": 512}))
    )


def first_point(report):
    return report["benchmark"]["points"][0]


def first_rows(report):
    return report["speedup"]["points"][0]["rows"]


def test_a_stand_in_report_verifies_as_synthetic_before_and_after_json(report):
    summary = verify_report(report, config=CONFIG)

    assert verify_report(stand_in_report(), config=CONFIG) == summary
    assert summary["measured"] is False
    assert summary["scope"] == "synthetic_backend_or_clock"
    assert (summary["correctness_cases"], summary["benchmark_points"]) == (3, 2)
    assert summary["refused_variants"] == []
    assert summary["checklist_sha256"] == report["checklist"]["sha256"]
    assert summary["cases_with_other_bits"] == {
        name: [] for name in report["summary"]["kernels_checked"]
    }
    assert len(summary["best"]) == 2
    other = config_from_mapping({**make_config_mapping(), "seed": 78})
    with pytest.raises(MatmulChecklistVerificationError, match="supplied config"):
        verify_report(report, config=other)


def test_measured_other_bits_and_refusals_verify_as_recorded(drifted, refused):
    summary = verify_report(drifted)
    outputs = drifted["correctness"]["cases"][1]["outputs"]
    kernels = drifted["summary"]["kernels_checked"]
    position = kernels.index("matmul_s5_pad4")

    assert summary["cases_with_other_bits"]["matmul_s5_pad4"] == [1, 2]
    assert outputs[position]["replay_bits"] is False
    assert outputs[position]["differing_elements"] == 1
    assert verify_report(refused)["refused_variants"] == ["matmul_s6_group_32x32"]


def test_an_unsealed_change_fails_on_the_digest(report):
    changed = copy.deepcopy(report)
    changed["summary"]["warnings"] = 1

    with pytest.raises(MatmulChecklistVerificationError, match="digest does not match"):
        verify_report(changed)


def set_sample(report):
    first_point(report)["sides"]["matmul_x_columns"]["samples_ns"][0] += 1


def set_median(report):
    first_point(report)["sides"]["matmul_s4_vec4"]["timing"]["median_ns"] += 1


def set_verdict(report):
    row = first_rows(report)[2]
    row["to_base"]["verdict"] = (
        "slower" if row["to_base"]["verdict"] != "slower" else ("faster")
    )


def set_held(report):
    row = first_rows(report)[3]
    row["held"] = not row["held"]


def set_fraction(report):
    first_rows(report)[4]["fraction_of_torch"] = 0.5


def set_best(report):
    report["speedup"]["points"][1]["best"]["variant"] = "matmul_x_rows"


def set_prediction(report):
    first_rows(report)[5]["predicted"] = ["faster"]


def set_hypothesis_record(report):
    record = report["speedup"]["hypotheses"][0]
    record["held_at"], record["missed_at"] = record["missed_at"], record["held_at"]


def set_cublas(report):
    report["speedup"]["cublas"] = "measured with cuBLAS"


def set_vendor(report):
    report["speedup"]["vendor"] = "vendor GPU matmul: cuBLAS"


def set_boundary(report):
    report["evidence_boundary"]["hardware_counters_used"] = True


def set_measured(report):
    report["measured"] = True


def set_hypothesis_text(report):
    report["checklist"]["steps"][4]["hypothesis"] = "S4 is faster than S3."


def set_checklist_digest(report):
    report["checklist"]["sha256"] = "0" * 64


def set_variant_digest(report):
    report["shaders"]["variants"][4]["source_sha256"] = "0" * 64


def set_library_digest(report):
    report["shaders"]["library_sha256"] = "e" * 64


def set_naive_digest(report):
    report["shaders"]["naive_source_sha256"] = "f" * 64


def set_limits(report):
    report["shaders"]["variants"][8]["limits"]["max_threads_per_threadgroup"] = 512


def set_admission(report):
    report["shaders"]["variants"][0]["admission"] = report["shaders"]["variants"][2][
        "admission"
    ]


def set_guard_digest(report):
    report["correctness"]["cases"][1]["outputs"][1]["buffer_sha256"] = "a" * 64


def set_guard_count(report):
    report["correctness"]["cases"][2]["outputs"][9]["guard_elements"] += 1


def set_replay_bits(report):
    report["correctness"]["cases"][1]["outputs"][3]["replay_bits"] = False


def set_float_count(report):
    report["correctness"]["cases"][2]["outputs"][0]["differing_elements"] = 0.0


def set_check(report):
    first_point(report)["checks"]["torch_matmul"]["disagreement"] = 1.0


def set_load(report):
    report["host_load"]["after"][1] = -1.0


def set_warning_count(report):
    report["warnings"]["count"] = 2


def set_switch(report):
    report["torch_mps_switches"]["values"]["PYTORCH_MPS_PREFER_METAL"] = "/Users/x"


def set_environment(report):
    report["environment"]["macos_version"] = "27 2"


def set_planned(report):
    report["planned_work"]["calls"] += 1


def set_elapsed(report):
    report["run"]["elapsed_ns"] = report["run"]["timed_ns"] - 1


def set_summary(report):
    report["summary"]["max_bound_ratio"] = 0.0


def set_summary_hypotheses(report):
    report["summary"]["hypotheses"]["matmul_s7_interior"]["held"] += 1


def set_config(report):
    report["config"]["seed"] = 78


def set_sides(report):
    first_point(report)["sides"] = []


def set_field(report):
    report["notes"] = "faster than cuBLAS"


def set_model(report):
    first_point(report)["model"]["matmul_s3_block_4x4"]["loads_per_output"] = 1.0


def set_order(report):
    first_point(report)["side_order"].reverse()


def set_same_bits(report):
    # Equal bits leave no difference, but the stand-in torch side sits 2**-30 away.
    first_point(report)["checks"]["torch_matmul"]["same_bits_as_naive"] = True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (set_sample, "point 0 .* does not recompute"),
        (set_median, "point 0 .* does not recompute"),
        (set_verdict, "speedup rows or the hypotheses"),
        (set_held, "speedup rows or the hypotheses"),
        (set_fraction, "speedup rows or the hypotheses"),
        (set_best, "speedup rows or the hypotheses"),
        (set_prediction, "speedup rows or the hypotheses"),
        (set_hypothesis_record, "speedup rows or the hypotheses"),
        (set_cublas, "speedup rows or the hypotheses"),
        (set_vendor, "speedup rows or the hypotheses"),
        (set_boundary, "boundary were weakened"),
        (set_measured, "only the host timer"),
        (set_hypothesis_text, "declared checklist and its hypotheses"),
        (set_checklist_digest, "declared checklist and its hypotheses"),
        (set_variant_digest, "shader section does not recompute"),
        (set_library_digest, "shader section does not recompute"),
        (set_naive_digest, "shader section does not recompute"),
        (set_limits, "shader section does not recompute"),
        (set_admission, "shader section does not recompute"),
        (set_guard_digest, "does not rebuild from the replay"),
        (set_guard_count, "does not rebuild from the replay"),
        (set_replay_bits, "does not rebuild from the replay"),
        (set_float_count, "must count its differing elements"),
        (set_check, "within the agreement bound"),
        (set_load, "host_load"),
        (set_warning_count, "warning counts do not reconcile"),
        (set_switch, "MPS switches"),
        (set_environment, "environment macos_version"),
        (set_planned, "planned work"),
        (set_elapsed, "run time does not cover"),
        (set_summary, "summary does not reconcile"),
        (set_summary_hypotheses, "summary does not reconcile"),
        (set_config, "config digest does not match"),
        (set_sides, "sides must be an object"),
        (set_field, "fields do not match the schema"),
        (set_model, "point 0 .* does not recompute"),
        (set_order, "point 0 .* does not recompute"),
        (set_same_bits, "cannot hold the naive bits"),
    ],
)
def test_every_resealed_tamper_is_caught_by_recomputation(report, mutate, message):
    with pytest.raises(MatmulChecklistVerificationError, match=message):
        verify_report(resealed(report, mutate))


def test_a_drifted_record_cannot_claim_the_replay_or_a_larger_error(drifted):
    position = drifted["summary"]["kernels_checked"].index("matmul_s5_pad4")

    def claim_replay(report):
        output = report["correctness"]["cases"][1]["outputs"][position]
        output["differing_elements"] = 0
        output["replay_bits"] = True

    def past_bound(report):
        output = report["correctness"]["cases"][1]["outputs"][position]
        output["max_relative_error"] = 1.0
        output["bound_ratio"] = 1.0 / report["correctness"]["cases"][1]["bound"]

    with pytest.raises(MatmulChecklistVerificationError, match="does not rebuild"):
        verify_report(resealed(drifted, claim_replay))
    with pytest.raises(MatmulChecklistVerificationError, match="records other bits"):
        verify_report(resealed(drifted, past_bound))


def test_a_refusal_cannot_be_hidden_or_invented(refused, report):
    def hide(changed):
        changed["refusals"] = []

    def invent(changed):
        changed["refusals"] = copy.deepcopy(refused["refusals"])

    with pytest.raises(MatmulChecklistVerificationError, match="refusals do not match"):
        verify_report(resealed(refused, hide))
    with pytest.raises(MatmulChecklistVerificationError, match="refusals do not match"):
        verify_report(resealed(report, invent))


def test_the_verifier_runs_without_torch(tmp_path, report):
    path = tmp_path / "report.json"
    path.write_bytes(canonical_json_bytes(report) + b"\n")
    script = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.matmul_checklist_verify import verify_report\n"
        f"summary = verify_report(json.loads(open({str(path)!r}).read()))\n"
        "print(summary['report_sha256'], sys.modules['torch'] is None)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == [report["report_sha256"], "True"]
