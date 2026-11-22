"""Tests of the sealed comparison report and of its independent verifier.

The reports here come from the NumPy stand-in of the interpreter, so they are
synthetic. Each tamper test changes one field, seals the report again, and
expects the verifier to name the section that no longer reproduces.
"""

import copy
import json
import os
import re
import time

import numpy
import pytest

from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.triton_matmul_cli import EXIT_REJECTED, main
from ai_journey.triton_matmul_config import PINNED_PLAN, TritonMatmulPlan
from ai_journey.triton_matmul_kernel import (
    AUTOTUNE_CONFIGS,
    TritonMatmulError,
    source_sha256,
)
from ai_journey.triton_matmul_report import (
    DIGEST_FIELD,
    INTERPRETER_BOUNDARY,
    INTERPRETER_LABEL,
    SYNTHETIC_BOUNDARY,
    SYNTHETIC_LABEL,
    build_report,
    seal_report,
)
from ai_journey.triton_matmul_runtime import InterpreterBackend
from ai_journey.triton_matmul_verify import (
    TritonMatmulVerificationError,
    _recount_c,
    load_report,
    verify_report,
)
from test_triton_matmul_correctness import StandInBackend
from test_triton_matmul_recorded import RECORDED_REPORT_JSON

SMALL = TritonMatmulPlan(
    name="triton-matmul-small",
    seed=74,
    shapes=(MatmulShape(17, 33, 9), MatmulShape(31, 47, 129), MatmulShape(1, 1, 1)),
    configs=AUTOTUNE_CONFIGS[3:5],
)
CLOCK = 1795300000.0
LOADS = (1.25, 2.5, 3.75)


def build(plan=SMALL, backend=None):
    return build_report(
        plan,
        backend or StandInBackend(),
        clock=lambda: CLOCK,
        load_average=lambda: LOADS,
    )


@pytest.fixture(scope="module")
def small():
    return seal_report(build())


def resealed(report, change):
    copied = copy.deepcopy(report)
    copied.pop(DIGEST_FIELD)
    change(copied)
    return seal_report(copied)


def keys(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{prefix}.{key}"
            yield from keys(item, f"{prefix}.{key}")
    elif isinstance(value, list):
        for item in value:
            yield from keys(item, prefix)


def test_a_stand_in_report_verifies_as_synthetic_with_its_sections(small):
    assert verify_report(small, plan=SMALL) == SMALL
    assert small["label"] == SYNTHETIC_LABEL
    assert small["backend"] == "injected"
    assert small["evidence_boundary"] == dict(SYNTHETIC_BOUNDARY)
    assert small["evidence_boundary"]["triton_interpreter_on_cpu"] is False
    assert small["recorded_at"] == "2026-11-21T22:26:40Z"
    assert small["host_load"] == {"before": list(LOADS), "after": list(LOADS)}
    assert small["summary"] == {
        "cases": 3,
        "configs": 2,
        "runs": 12,
        "normal_runs_within_bound": 6,
        "exact_runs_with_exact_bits": 6,
        "max_bound_ratio": small["summary"]["max_bound_ratio"],
        "normal_elements_off_the_rounded_reference": (
            small["summary"]["normal_elements_off_the_rounded_reference"]
        ),
        "interpreter_steps": 52,
        "loc_texts": 19,
        "triton_kernel_lines": 33,
        "triton_host_lines": 9,
    }
    assert 0 < small["summary"]["max_bound_ratio"] < 1
    assert small["kernel"]["autotune"]["tuned"] is False
    assert len(small["kernel"]["autotune"]["configs"]) == 2


def test_the_report_holds_no_time_and_gflops_only_in_the_metal_rates(small):
    found = set(keys(small))

    assert not [key for key in found if re.search(r"seconds|duration|elapsed", key)]
    rates = {key for key in found if "gflops" in key}
    assert rates == {
        ".performance.shapes.metal.median_gflops_per_second",
        ".performance.shapes.metal.iqr_gflops_per_second",
    }
    assert small["performance"]["interpreter_time"].startswith("not reported")


def test_only_the_interpreter_backend_gets_the_interpreter_label():
    impostor = object.__new__(InterpreterBackend)
    impostor.source_sha256 = "0" * 64
    impostor.environment = dict(StandInBackend().environment)

    with pytest.raises(TritonMatmulError, match="did not compile the pinned text"):
        build(backend=impostor)
    assert INTERPRETER_BOUNDARY["triton_interpreter_on_cpu"] is True
    relabeled = resealed(
        seal_report(build()), lambda r: r.update(label=INTERPRETER_LABEL)
    )
    with pytest.raises(TritonMatmulVerificationError, match="backend does not match"):
        verify_report(relabeled)


def pinned_impostor():
    """An InterpreterBackend shell that runs the stand-in, to reach the label."""

    impostor = object.__new__(InterpreterBackend)
    impostor.source_sha256 = source_sha256()
    impostor.environment = dict(StandInBackend().environment)
    impostor.run = StandInBackend().run
    return impostor


def test_an_interpreter_check_takes_only_the_host_clock_and_load_average():
    # An interpreter label must not carry an injected time or load average.
    for clock, load_average in (
        (lambda: CLOCK, os.getloadavg),
        (time.time, lambda: LOADS),
    ):
        with pytest.raises(TritonMatmulError, match="host clock"):
            build_report(
                SMALL, pinned_impostor(), clock=clock, load_average=load_average
            )
    report = build_report(
        SMALL, pinned_impostor(), clock=time.time, load_average=os.getloadavg
    )
    assert report["label"] == INTERPRETER_LABEL


def test_a_backend_must_name_its_environment():
    backend = StandInBackend()
    backend.environment = {"python": "3.11.0"}

    with pytest.raises(TritonMatmulError, match="must name its environment"):
        build(backend=backend)
    with pytest.raises(TypeError, match="TritonMatmulPlan"):
        build(plan=SMALL.to_mapping())


def exact(report, case=0, run=1):
    return report["correctness"]["cases"][case]["runs"][run]


def normal(report, case=0, run=0):
    return report["correctness"]["cases"][case]["runs"][run]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["evidence_boundary"].update(gpu_executed=True), "evidence_boundary"),
        (lambda r: r["evidence_boundary"].update(autotuned=True), "evidence_boundary"),
        (lambda r: r.update(backend="triton_interpreter"), "backend does not match"),
        (lambda r: r.update(label="measured"), "known label"),
        (lambda r: r["kernel"].update(text="import triton\n"), "kernel"),
        (lambda r: r["kernel"].update(input_precision="tf32"), "kernel"),
        (lambda r: r["kernel"]["autotune"].update(tuned=True), "kernel"),
        (lambda r: r["kernel"]["autotune"]["configs"][0].update(tile_bytes=1), "kernel"),
        (lambda r: r["rule"].update(order="k order"), "rule"),
        (lambda r: r["rule"].update(sentinel_float32_bits=0), "rule"),
        (lambda r: r["loc"]["texts"][0]["lines"].update(kernel=20), "loc"),
        (lambda r: r["loc"]["texts"][3]["lines"].update(host=None), "loc"),
        (lambda r: r["loc"]["texts"][5].update(sha256="0" * 64), "loc"),
        (lambda r: r["performance"]["shapes"][0]["metal"][0].update(median_gflops_per_second=400.0), "performance"),
        (lambda r: r["performance"]["shapes"][2].update(triton_gpu=1000.0), "performance"),
        (lambda r: r["performance"]["metal_reports"][1].update(report_sha256="0" * 64), "performance"),
        (lambda r: r["ci_install"]["wheels"][0].update(sha256="0" * 64), "ci_install"),
        (lambda r: exact(r).update(buffer_sha256="0" * 64), r"cases\[0\].runs\[1\]"),
        (lambda r: exact(r, 1, 3).update(max_relative_error=1e-9), r"cases\[1\].runs\[3\]"),
        (lambda r: exact(r, 2).update(guard_elements=0), r"cases\[2\].runs\[1\]"),
        (lambda r: normal(r).update(max_relative_error=1.0), "no error within the bound"),
        (lambda r: normal(r).update(bound_ratio=0.5), "ratio that does not recompute"),
        (lambda r: normal(r, 1).update(rounded_reference_differences=0), "disagrees with its digest"),
        (lambda r: normal(r, 1).update(programs=7), r"cases\[1\].runs\[0\]"),
        (lambda r: normal(r, 1, 2).update(k_blocks=1), r"cases\[1\].runs\[2\]"),
        (lambda r: r["correctness"]["cases"][1]["operands"][0].update(a_sha256="0" * 64), r"cases\[1\]"),
        (lambda r: r["correctness"]["cases"][0].update(error_bound=1.0), r"cases\[0\]"),
        (lambda r: r["correctness"]["cases"][0]["runs"].pop(), "runs do not match"),
        (lambda r: r["correctness"]["cases"].pop(), "cases do not match the plan"),
        (lambda r: r["summary"].update(max_bound_ratio=0.0), "summary"),
        (lambda r: r["summary"].update(loc_texts=18), "summary"),
        (lambda r: r["environment"].update(triton="3.9.0 interpreted"), "environment"),
        (lambda r: r.update(recorded_at="2026-11-22 10:00"), "recorded_at"),
        (lambda r: r["host_load"].update(before=[1, 2, 3]), "host_load"),
        (lambda r: r.update(plan_sha256="0" * 64), "plan_sha256"),
    ],
)  # fmt: skip
def test_a_resealed_change_to_any_section_is_named(small, change, message):
    with pytest.raises(TritonMatmulVerificationError, match=message):
        verify_report(resealed(small, change), plan=SMALL)


def test_the_seal_the_schema_and_the_plan_binding_are_checked(small):
    unsealed = copy.deepcopy(small)
    unsealed["summary"]["cases"] = 4
    with pytest.raises(TritonMatmulVerificationError, match="digest does not match"):
        verify_report(unsealed)
    with pytest.raises(TritonMatmulVerificationError, match="fields do not match"):
        verify_report({**small, "extra": 1})
    with pytest.raises(TritonMatmulVerificationError, match="not built from this plan"):
        verify_report(small, plan=PINNED_PLAN)
    renamed = resealed(small, lambda r: r["plan"].update(name="Other Plan"))
    with pytest.raises(TritonMatmulVerificationError, match="embedded plan is invalid"):
        verify_report(renamed)
    kind = resealed(small, lambda r: r.update(kind="gpu_tiled_cuda"))
    with pytest.raises(TritonMatmulVerificationError, match="kind"):
        verify_report(kind)
    with pytest.raises(TritonMatmulError, match="already sealed"):
        seal_report(small)


def test_the_verifier_rebuilds_each_exact_buffer_on_the_cpu(small):
    # An exact output that is one ulp off fails, although it keeps the bound.
    def nudge(report):
        record = exact(report)
        record["buffer_sha256"] = record["buffer_sha256"][::-1]

    with pytest.raises(TritonMatmulVerificationError, match=r"runs\[1\].buffer_sha256"):
        verify_report(resealed(small, nudge))
    assert numpy.float32(1.0) != numpy.nextafter(numpy.float32(1.0), numpy.float32(2))


def set_normal_errors(report, case, error):
    """Give each normal run of a case another error and keep its sums consistent."""

    record = report["correctness"]["cases"][case]
    for run in record["runs"]:
        if run["family"] == "normal":
            run["max_relative_error"] = error
            run["bound_ratio"] = error / record["error_bound"]
    report["summary"]["max_bound_ratio"] = max(
        run["bound_ratio"]
        for item in report["correctness"]["cases"]
        for run in item["runs"]
        if run["family"] == "normal"
    )


def test_a_normal_record_with_the_rounded_bits_keeps_their_error(small):
    # At 1 x 1 x 1 the output is the rounded reference, so its error is known.
    case = SMALL.shapes.index(MatmulShape(1, 1, 1))
    run = normal(small, case)
    bound = small["correctness"]["cases"][case]["error_bound"]
    assert run["rounded_reference_differences"] == 0 < run["max_relative_error"]

    higher = (run["max_relative_error"] + bound) / 2
    forged = resealed(small, lambda r: set_normal_errors(r, case, higher))
    with pytest.raises(TritonMatmulVerificationError, match="bits of the rounded"):
        verify_report(forged, plan=SMALL)


def test_a_normal_error_below_the_rounded_reference_error_is_refused(small):
    # No float32 output is nearer the float64 sum than its rounding.
    forged = resealed(small, lambda r: set_normal_errors(r, 1, 1e-12))

    with pytest.raises(
        TritonMatmulVerificationError, match="below the error of the rounded"
    ):
        verify_report(forged, plan=SMALL)


def test_the_recorded_largest_bound_ratio_cannot_be_lowered():
    # The largest ratio, 0.516, is the 1 x 1 x 1 case, whose output bits are known.
    recorded = json.loads(RECORDED_REPORT_JSON)
    assert recorded["summary"]["max_bound_ratio"] == normal(recorded)["bound_ratio"]
    assert normal(recorded)["rounded_reference_differences"] == 0

    forged = resealed(recorded, lambda r: set_normal_errors(r, 0, 1e-9))
    assert forged["summary"]["max_bound_ratio"] < 0.25
    with pytest.raises(TritonMatmulVerificationError, match=r"cases\[0\].runs\[0\]"):
        verify_report(forged, plan=PINNED_PLAN)


def test_a_label_that_is_not_a_string_is_rejected_with_exit_code_1(small, tmp_path):
    relabeled = resealed(small, lambda r: r.update(label=[SYNTHETIC_LABEL]))
    path = tmp_path / "relabeled.json"
    path.write_bytes(canonical_json_bytes(relabeled) + b"\n")

    with pytest.raises(TritonMatmulVerificationError, match="known label"):
        verify_report(relabeled)
    assert main(["verify", "--report", str(path)]) == EXIT_REJECTED


def test_a_report_file_must_hold_canonical_bytes(small, tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(canonical_json_bytes(small) + b"\n")
    assert load_report(path, plan=SMALL) == small

    path.write_bytes(json.dumps(small, indent=1).encode("ascii"))
    with pytest.raises(TritonMatmulVerificationError, match="not canonical"):
        load_report(path)
    path.write_bytes(b"[1]\n")
    with pytest.raises(TritonMatmulVerificationError, match="JSON object"):
        load_report(path)
    path.write_bytes(b"{")
    with pytest.raises(TritonMatmulVerificationError, match="strict UTF-8 JSON"):
        load_report(path)
    with pytest.raises(TritonMatmulError, match="readable regular") as raised:
        load_report(tmp_path / "missing.json")
    assert type(raised.value) is TritonMatmulError


def test_the_second_counter_refuses_a_text_outside_its_rule():
    with pytest.raises(TritonMatmulVerificationError, match="block comment"):
        _recount_c("/* x */\n__global__ void k() {\n}\n", metal=False)
    with pytest.raises(TritonMatmulVerificationError, match="outside its kernels"):
        _recount_c("kernel void k() {\n}\nvoid host() {\n}\n", metal=True)
