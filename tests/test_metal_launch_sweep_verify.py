import copy
import functools
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import report_digest
from ai_journey.metal_launch_sweep_config import config_from_mapping
from ai_journey.metal_launch_sweep_correctness import matmul_inputs
from ai_journey.metal_launch_sweep_report import MEASUREMENT_BOUNDARY, measure_report
from ai_journey.metal_launch_sweep_verify import (
    MetalLaunchSweepVerificationError,
    verify_report,
)
from test_kernel_metal_runtime import requires_metal
from test_metal_launch_sweep_config import changed, make_config_mapping
from test_metal_launch_sweep_standin import StandInBackend

ROOT = Path(__file__).resolve().parents[1]


@functools.cache
def sealed_json():
    config = config_from_mapping(make_config_mapping())
    clock = itertools.count(0, 1_000).__next__
    report = measure_report(config, backend=StandInBackend(), clock=clock)
    return canonical_json_bytes(report).decode("ascii")


def fresh():
    """Return a stand-in report as a verifier reads it from a file."""

    return json.loads(sealed_json())


def reseal(report):
    report["report_sha256"] = report_digest(report)
    return report


def claim_measured(report):
    report.update(
        measured=True,
        timer="time.perf_counter_ns",
        backend="torch.mps.compile_shader",
        measurement_scope="apple_gpu_metal_this_host",
        label="host-specific Apple GPU (Metal) measurement",
        evidence_boundary=copy.deepcopy(MEASUREMENT_BOUNDARY),
    )
    report["environment"]["platform_system"] = "Darwin"
    report["summary"]["label"] = report["label"]


def test_a_stand_in_report_verifies_as_synthetic_and_binds_its_config():
    report = fresh()
    config = config_from_mapping(make_config_mapping())

    summary = verify_report(report, config=config)

    assert summary["measured"] is False
    assert summary["scope"] == "synthetic_backend_or_clock"
    assert summary["report_sha256"] == report["report_sha256"]
    assert (summary["admitted_shapes"], summary["rejected_shapes"]) == (7, 0)
    assert (summary["correctness_cases"], summary["benchmark_points"]) == (6, 4)
    other = config_from_mapping(changed(("seed",), 75))
    with pytest.raises(MetalLaunchSweepVerificationError, match="supplied config"):
        verify_report(report, config=other)
    with pytest.raises(TypeError, match="MetalLaunchSweepConfig"):
        verify_report(report, config=make_config_mapping())


def tamper_limits(report):
    report["pipelines"]["matmul_naive"]["max_threads_per_threadgroup"] = 128


def tamper_rule(report):
    report["dispatch"]["requires_whole_simd_groups"] = True


def tamper_footprint(report):
    report["admission"]["matmul"][2]["rows_per_simd_group"] = 1


def tamper_mapping(report):
    report["mapping"]["checks"][0]["write_conflicts"] = 1


def tamper_layout(report):
    report["simd_layout"][3]["observed_sha256"] = "0" * 64


def tamper_case_input(report):
    report["correctness"]["matmul"][1]["a_sha256"] = "0" * 64


def tamper_case_value(report):
    sample = report["correctness"]["matmul"][1]["samples"][6]
    sample[2] = sample[2] + 1.0


def tamper_case_ratio(report):
    report["correctness"]["matmul"][1]["max_bound_ratio"] = 0.0


def tamper_case_guard(report):
    report["correctness"]["matmul"][0]["shapes"][0]["guard_elements"] -= 1


def tamper_vector_case(report):
    report["correctness"]["vector_add"][2]["outputs"][0]["guard_elements"] = 0


def tamper_sample(report):
    report["benchmark"]["matmul"][0]["shapes"][0]["samples_ns"][0] += 1


def tamper_ranking(report):
    point = report["benchmark"]["vector"][1]
    point["ranking"] = point["ranking"][::-1]


def tamper_point_value(report):
    sample = report["benchmark"]["matmul"][1]["accuracy"]["samples"][0]
    sample[2] = -sample[2] - 1.0


def tamper_vector_output(report):
    report["benchmark"]["vector"][0]["output_sha256"] = "0" * 64


def tamper_planned(report):
    report["planned_work"]["calls"] -= 1


def tamper_elapsed(report):
    report["run"]["elapsed_ns"] = report["run"]["timed_ns"] - 1


def tamper_load(report):
    report["run"]["load_average_after"] = [1, 2, 3]


def tamper_summary(report):
    report["summary"]["best_vector_gb_per_second"] *= 2


def tamper_boundary(report):
    report["evidence_boundary"]["vendor_peak_claim"] = True


def tamper_source(report):
    report["shaders"]["source"] = report["shaders"]["source"].replace("+=", "-=")


def tamper_measured(report):
    report["measured"] = True


def tamper_measured_rule(report):
    claim_measured(report)
    tamper_rule(report)


def tamper_extra_field(report):
    report["notes"] = "faster than a CUDA GPU"


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (tamper_limits, "admission does not follow from the recorded limits"),
        (tamper_rule, "admission does not follow from the recorded limits"),
        (tamper_footprint, "admission does not follow"),
        (tamper_mapping, "mapping checks do not rebuild"),
        (tamper_layout, "SIMD layout records do not rebuild"),
        (tamper_case_input, "matmul case 1 does not rebuild from the seed"),
        (tamper_case_value, "exceeds the bound"),
        (tamper_case_ratio, "error maxima do not cover the lattice"),
        (tamper_case_guard, "matmul case 0 does not rebuild"),
        (tamper_vector_case, "vector_add case 2"),
        (tamper_sample, "does not recompute from its samples"),
        (tamper_ranking, "does not recompute from its samples"),
        (tamper_point_value, "matmul point 1: C\\[0, 0\\] exceeds the bound"),
        (tamper_vector_output, "is not the float32 sum"),
        (tamper_planned, "planned work does not match"),
        (tamper_elapsed, "run time or load averages"),
        (tamper_load, "run time or load averages"),
        (tamper_summary, "summary does not reconcile"),
        (tamper_boundary, "labels or boundary were weakened"),
        (tamper_source, "this lab's Metal sources"),
        (tamper_measured, "only the host timer"),
        (tamper_measured_rule, "do not require whole SIMD groups"),
        (tamper_extra_field, "report fields do not match"),
    ],
)
def test_a_resealed_report_with_any_changed_fact_is_rejected(tamper, message):
    report = fresh()
    tamper(report)
    reseal(report)

    with pytest.raises(MetalLaunchSweepVerificationError, match=message):
        verify_report(report)


def test_the_error_maxima_of_a_case_cannot_fall_below_its_lattice():
    case = fresh()["correctness"]["matmul"][0]
    # The 1 x 1 x 1 case holds its whole C on the lattice, and its error is
    # below 1e-9, so a fixed slack of that size would let its maximum drop to 0.
    assert (case["m"], case["n"], case["k"]) == (1, 1, 1)
    assert 0 < case["max_abs_error"] < 1e-9

    for value in (0.0, -case["max_abs_error"], case["max_abs_error"] / 2):
        report = fresh()
        report["correctness"]["matmul"][0]["max_abs_error"] = value
        with pytest.raises(MetalLaunchSweepVerificationError, match="error maxima"):
            verify_report(reseal(report))


def float64_value(seed, index, shape, row, column):
    """Return one element of the float64 product, as no float32 kernel holds it."""

    a, b = (
        matrix.astype(numpy.float64) for matrix in matmul_inputs(seed, index, *shape)
    )
    value = float(a[row] @ b[:, column])
    assert float(numpy.float32(value)) != value
    return value


def test_lattice_values_of_c_must_be_float32_values():
    # Each changed value meets the bound with no error at all, but C is a
    # float32 buffer, so no kernel can return it.
    report = fresh()
    case = report["correctness"]["matmul"][0]
    assert (case["m"], case["n"], case["k"]) == (1, 1, 1)
    case["samples"][0][2] = float64_value(74, 0, (1, 1, 1), 0, 0)
    with pytest.raises(MetalLaunchSweepVerificationError, match="float32 lattice"):
        verify_report(reseal(report))

    report = fresh()
    point = report["benchmark"]["matmul"][0]
    sample = point["accuracy"]["samples"][3]
    shape = (point["m"], point["n"], point["k"])
    sample[2] = float64_value(74, 32, shape, sample[0], sample[1])
    with pytest.raises(MetalLaunchSweepVerificationError, match="float32 lattice"):
        verify_report(reseal(report))


def test_an_unsealed_change_or_a_malformed_section_is_rejected():
    report = fresh()
    report["summary"]["rejected_shapes"] = 1
    with pytest.raises(MetalLaunchSweepVerificationError, match="digest"):
        verify_report(report)
    for section, value in (
        ("pipelines", []),
        ("benchmark", {"matmul": None}),
        ("correctness", "none"),
        ("run", None),
    ):
        broken = fresh()
        broken[section] = value
        with pytest.raises(MetalLaunchSweepVerificationError):
            verify_report(reseal(broken))
    with pytest.raises(MetalLaunchSweepVerificationError, match="JSON object"):
        verify_report([])


def test_verification_never_imports_torch(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(sealed_json(), encoding="ascii")
    script = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.metal_launch_sweep_verify import verify_report\n"
        f"report = json.loads(open({str(path)!r}).read())\n"
        "print(verify_report(report)['scope'])\n"
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
    assert completed.stdout.strip() == "synthetic_backend_or_clock"


@requires_metal
def test_a_small_measured_report_from_this_gpu_verifies_as_a_measurement():
    config = config_from_mapping(make_config_mapping())

    report = json.loads(canonical_json_bytes(measure_report(config)))
    summary = verify_report(report, config=config)

    assert summary["measured"] is True
    assert summary["scope"] == "apple_gpu_metal_this_host"
    assert summary["label"] == "host-specific Apple GPU (Metal) measurement"
