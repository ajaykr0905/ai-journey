"""Tests of the sealed Day 81 report and its independent CPU verifier."""

import base64
import copy
import hashlib

import numpy
import pytest

from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.triton_flash_config import PINNED_PLAN, TritonFlashPlan
from ai_journey.triton_flash_kernel import FLASH_CONFIGS, TritonFlashError
from ai_journey.triton_flash_reference import AttentionCase
from ai_journey.triton_flash_report import (
    DIGEST_FIELD,
    INTERPRETER_BOUNDARY,
    INTERPRETER_LABEL,
    SYNTHETIC_LABEL,
    build_report,
    report_digest,
    seal_report,
)
from ai_journey.triton_flash_verify import (
    TritonFlashVerificationError,
    load_report,
    verify_report,
)
from test_triton_flash_correctness import StandInBackend

SMALL_PLAN = TritonFlashPlan(
    name="small-flash",
    seed=81,
    cases=(AttentionCase(2, 17, 16, True), AttentionCase(1, 40, 32, False)),
    configs=(FLASH_CONFIGS[3],),
)


def fixed_clock():
    return 1795478400.0


def fixed_load():
    return (1.5, 1.25, 1.0)


def build(plan=SMALL_PLAN):
    return seal_report(
        build_report(plan, StandInBackend(), clock=fixed_clock, load_average=fixed_load)
    )


@pytest.fixture(scope="module")
def sealed():
    return build()


def reseal(report):
    unsealed = {key: value for key, value in report.items() if key != DIGEST_FIELD}
    return seal_report(unsealed)


def test_a_stand_in_report_is_synthetic_and_verifies(sealed):
    plan = verify_report(sealed, plan=SMALL_PLAN)

    assert plan == SMALL_PLAN
    assert sealed["label"] == SYNTHETIC_LABEL
    assert sealed["backend"] == "injected"
    assert sealed["evidence_boundary"]["triton_interpreter_on_cpu"] is False
    assert sealed["evidence_boundary"]["gpu_executed"] is False
    assert sealed["evidence_boundary"]["memory_measured"] is False
    assert sealed["recorded_at"] == "2026-11-24T00:00:00Z"
    assert sealed["host_load"] == {
        "before": [1.5, 1.25, 1.0],
        "after": [1.5, 1.25, 1.0],
    }
    summary = sealed["summary"]
    assert (summary["cases"], summary["runs"], summary["runs_within_bound"]) == (
        2,
        4,
        4,
    )
    assert summary["causal_runs"] == 2
    assert 0 < summary["max_output_bound_ratio"] < 0.05
    assert sealed["memory"]["measured"] is False
    assert sealed[DIGEST_FIELD] == report_digest(sealed)


def test_the_pinned_plan_report_verifies_with_its_plan():
    report = build(PINNED_PLAN)

    verify_report(report, plan=PINNED_PLAN)
    assert report["summary"]["runs"] == 48
    assert len(canonical_json_bytes(report)) < 4 * 1024 * 1024


def test_load_report_needs_canonical_bytes(tmp_path, sealed):
    path = tmp_path / "report.json"
    path.write_bytes(canonical_json_bytes(sealed) + b"\n")
    assert load_report(path, plan=SMALL_PLAN)["report_sha256"] == sealed[DIGEST_FIELD]

    path.write_bytes(canonical_json_bytes(sealed))
    with pytest.raises(TritonFlashVerificationError, match="canonical"):
        load_report(path)
    with pytest.raises(TritonFlashError, match="report"):
        load_report(tmp_path / "missing.json")


def first_run(report):
    return report["correctness"]["cases"][0]["runs"][0]


def shift_output(report, amount):
    run = first_run(report)
    values = numpy.frombuffer(base64.b64decode(run["output"]), dtype="<f4").copy()
    values[5] += numpy.float32(amount)
    run["output"] = base64.b64encode(values.tobytes()).decode("ascii")
    run["output_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()


def tamper_label(report):
    report["label"] = INTERPRETER_LABEL


def tamper_boundary(report):
    report["label"] = INTERPRETER_LABEL
    report["backend"] = "triton_interpreter"
    report["evidence_boundary"] = dict(INTERPRETER_BOUNDARY)


def tamper_gpu(report):
    report["evidence_boundary"]["gpu_executed"] = True


def tamper_text(report):
    report["kernel"]["text"] = report["kernel"]["text"].replace(
        "alpha = tl.exp(m_i - m_new)", "alpha = 1.0"
    )


def tamper_memory(report):
    report["memory"]["rows"][3]["naive_bytes"] += 4


def tamper_measured(report):
    report["memory"]["measured"] = True


def tamper_value_only(report):
    run = first_run(report)
    values = numpy.frombuffer(base64.b64decode(run["output"]), dtype="<f4").copy()
    values[0] = 0.5
    run["output"] = base64.b64encode(values.tobytes()).decode("ascii")


def tamper_value_out_of_bound(report):
    shift_output(report, 1.0)


def tamper_sum_below_one(report):
    run = first_run(report)
    values = numpy.frombuffer(base64.b64decode(run["sums"]), dtype="<f4").copy()
    values[0] = 0.5
    run["sums"] = base64.b64encode(values.tobytes()).decode("ascii")
    run["sums_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()


def tamper_error(report):
    first_run(report)["max_output_error"] *= 0.5


def tamper_bound(report):
    first_run(report)["output_bounds"][0] *= 4


def tamper_guard(report):
    first_run(report)["guard_rows"] = 0


def tamper_operand(report):
    report["correctness"]["cases"][0]["operands"][0]["q_sha256"] = "0" * 64


def tamper_summary(report):
    report["summary"]["runs_within_bound"] -= 1


def tamper_environment(report):
    report["environment"]["machine"] = "a b"


def tamper_case(report):
    report["correctness"]["cases"][0]["case"]["causal"] = False


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        (tamper_label, "backend does not match its label"),
        (tamper_boundary, "TRITON_INTERPRET=1"),
        (tamper_gpu, "evidence_boundary.gpu_executed"),
        (tamper_text, "kernel.sha256|kernel.text"),
        (tamper_memory, "memory.rows"),
        (tamper_measured, "unmeasured model"),
        (tamper_value_only, "output_sha256 does not match"),
        (tamper_value_out_of_bound, "breaks the output bound"),
        (tamper_sum_below_one, "running sum below 1"),
        (tamper_error, "max_output_error"),
        (tamper_bound, "output_bounds"),
        (tamper_guard, "guard_rows"),
        (tamper_operand, "operands"),
        (tamper_summary, "summary.runs_within_bound"),
        (tamper_environment, "environment"),
        (tamper_case, "case 0.case"),
    ),
)
def test_a_resealed_tampered_report_fails_verification(sealed, tamper, message):
    report = copy.deepcopy(sealed)
    tamper(report)

    with pytest.raises(TritonFlashVerificationError, match=message):
        verify_report(reseal(report))


def test_an_unresealed_change_breaks_the_seal(sealed):
    report = copy.deepcopy(sealed)
    shift_output(report, 1e-7)

    with pytest.raises(TritonFlashVerificationError, match="digest"):
        verify_report(report)


def test_a_report_from_another_plan_is_refused(sealed):
    with pytest.raises(TritonFlashVerificationError, match="not built from this plan"):
        verify_report(sealed, plan=PINNED_PLAN)


def test_the_report_refuses_bad_inputs():
    with pytest.raises(TypeError):
        build_report(
            PINNED_PLAN.to_mapping(),
            StandInBackend(),
            clock=fixed_clock,
            load_average=fixed_load,
        )
    backend = StandInBackend()
    backend.environment = {"python": "3.11"}
    with pytest.raises(TritonFlashError, match="environment"):
        build_report(SMALL_PLAN, backend, clock=fixed_clock, load_average=fixed_load)
    with pytest.raises(TritonFlashError, match="load average"):
        build_report(
            SMALL_PLAN,
            StandInBackend(),
            clock=fixed_clock,
            load_average=lambda: (1.0, float("nan"), 1.0),
        )
    with pytest.raises(TritonFlashError, match="already sealed"):
        seal_report(build())
    with pytest.raises(TritonFlashVerificationError, match="schema"):
        verify_report({"kind": "triton_flash_attention"})
