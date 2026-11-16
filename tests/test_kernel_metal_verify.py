import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

from ai_journey.kernel_metal_config import config_from_mapping
from ai_journey.kernel_metal_reference import (
    all_rgb_triples,
    array_sha256,
    rgb_to_gray_reference,
)
from ai_journey.kernel_metal_report import REPORT_DIGEST_FIELD, measure_report
from ai_journey.kernel_metal_verify import KernelMetalVerificationError, verify_report
from ai_journey.memory_access_report import seal_report
from test_kernel_metal_config import make_config_mapping
from test_kernel_metal_recorded import RECORDED_REPORT_JSON, RecordedBackend

ROOT = Path(__file__).resolve().parents[1]
RECORDED = json.loads(RECORDED_REPORT_JSON)
CONFIG = config_from_mapping(make_config_mapping())


def varied_clock():
    """Advance by a repeating mix of steps so samples and statistics vary."""

    steps = itertools.cycle((997, 1201, 1499, 1999, 2503, 1103, 1789))
    return itertools.accumulate(steps, initial=0).__next__


SYNTHETIC = measure_report(CONFIG, backend=RecordedBackend(), clock=varied_clock())


def fresh(report=RECORDED):
    return json.loads(json.dumps(report))


def reseal(report):
    unsigned = {
        key: value for key, value in report.items() if key != REPORT_DIGEST_FIELD
    }
    return seal_report(unsigned)


def tampered(path, change, report=RECORDED):
    copy = fresh(report)
    target = copy
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = change(target[path[-1]])
    return reseal(copy)


def test_measured_and_synthetic_reports_verify_and_bind_to_their_config():
    recorded = verify_report(fresh(), config=config_from_mapping(RECORDED["config"]))
    synthetic = verify_report(fresh(SYNTHETIC), config=CONFIG)

    assert recorded == {
        "kind": "kernel_metal_correctness_and_benchmark",
        "report_sha256": RECORDED[REPORT_DIGEST_FIELD],
        "config_sha256": RECORDED["config_sha256"],
        "measured": True,
        "scope": "apple_gpu_metal_this_host",
        "label": "host-specific Apple GPU (Metal) measurement",
        "correctness_cases": 4,
        "benchmark_points": 2,
        "timed_samples": 12,
        "rgb_to_gray_max_abs_difference_levels": 1,
        "rgb_to_gray_metal_mismatched_pixels": 21,
    }
    assert (synthetic["measured"], synthetic["scope"]) == (
        False,
        "synthetic_backend_or_clock",
    )
    assert synthetic["label"] == "synthetic timing, not a measurement"
    with pytest.raises(KernelMetalVerificationError, match="supplied config"):
        verify_report(fresh(), config=CONFIG)


def test_an_edit_without_a_new_seal_breaks_the_digest():
    report = fresh()
    report["benchmark"]["vector_add"][0]["metal_kernel"]["samples_ns"][0] += 1

    with pytest.raises(KernelMetalVerificationError, match="digest does not match"):
        verify_report(report)


TRIPLES = ("correctness", "rgb_to_gray", 1, "outputs", 0)
POINT = ("benchmark", "vector_add", 0, "metal_kernel")


@pytest.mark.parametrize(
    ("path", "change", "message"),
    [
        ((*TRIPLES, "mismatches"), lambda v: v[1:], "does not rebuild from its"),
        ((*TRIPLES, "mismatches"), lambda v: v + [[16777215, 0]], "does not rebuild"),
        ((*TRIPLES, "mismatches", 0, 1), lambda v: v - 1, "does not rebuild from its"),
        (
            (*TRIPLES, "mismatches", 0, 1),
            lambda v: v + 1,
            "mismatch matches the reference",
        ),
        ((*TRIPLES, "mismatches"), lambda v: v[::-1], "indices must increase"),
        ((*TRIPLES, "mismatched_pixels"), lambda v: v - 1, "does not rebuild"),
        ((*TRIPLES, "max_abs_difference_levels"), lambda v: 0, "does not rebuild"),
        ((*TRIPLES, "guard_elements"), lambda v: v + 1, "does not rebuild"),
        (
            ("correctness", "vector_add", 0, "outputs", 0, "buffer_sha256"),
            lambda v: "0" * 64,
            "metal_kernel on mps is not exact",
        ),
        (
            ("correctness", "vector_add", 1, "outputs", 2, "buffer_sha256"),
            lambda v: "0" * 64,
            "torch on cpu is not exact",
        ),
        (
            ("correctness", "vector_add", 0, "a_sha256"),
            lambda v: "0" * 64,
            "vector_add case 0 does not rebuild from the seed",
        ),
        (
            ("correctness", "rgb_to_gray", 0, "launch", "padding_threads"),
            lambda v: v - 1,
            "rgb_to_gray case 0 does not rebuild from the seed",
        ),
        (
            ("correctness", "rgb_to_gray", 0, "bound_levels"),
            lambda v: 2,
            "rgb_to_gray case 0 does not rebuild from the seed",
        ),
        (("correctness", "bound_levels"), lambda v: 2, "lab's reference and bound"),
        (("correctness", "rounding"), lambda v: "round half even", "reference and"),
        ((*POINT, "samples_ns", 0), lambda v: v + 1, "does not recompute from its"),
        ((*POINT, "samples_ns"), lambda v: v[:2], "hold 3 positive integers"),
        ((*POINT, "samples_ns", 1), lambda v: True, "hold 3 positive integers"),
        ((*POINT, "samples_ns", 1), lambda v: 0, "hold 3 positive integers"),
        ((*POINT, "calls_per_sample"), lambda v: v + 1, "does not recompute"),
        ((*POINT, "median_gb_per_second"), lambda v: v * 2, "does not recompute"),
        (
            ("benchmark", "rgb_to_gray", 0, "torch_to_metal_time_ratio"),
            lambda v: v * 2,
            "rgb_to_gray point at",
        ),
        (("benchmark", "synchronization"), lambda v: "none", "benchmark section"),
        (("measured",), lambda v: False, "only the host timer"),
        (("backend",), lambda v: "injected", "only the host timer"),
        (("timer",), lambda v: "injected", "only the host timer"),
        (("timer",), lambda v: "time.time", "method fields were changed"),
        (("label",), lambda v: "Apple GPU measurement", "weakened"),
        (("measurement_scope",), lambda v: "cuda_device", "weakened"),
        (("evidence_boundary", "cuda_kernel_executed"), lambda v: True, "weakened"),
        (("evidence_boundary", "statement"), lambda v: v[:40], "weakened"),
        (("environment", "torch_version"), lambda v: "2.15 custom", "token"),
        (("environment", "platform_system"), lambda v: "Linux", "from macOS"),
        (("environment",), lambda v: {**v, "hostname": "x"}, "host record"),
        (("shaders", "source"), lambda v: v.replace("i < n", "true"), "Metal source"),
        (
            ("shaders", "contraction_probe", "observed"),
            lambda v: 1.0,
            "fused or the unfused",
        ),
        (("shaders", "contraction_probe", "fused"), lambda v: False, "shader section"),
        (
            ("shaders", "kernels", 0, "limits", "max_threads_per_threadgroup"),
            lambda v: 128,
            "exceeds its pipeline's recorded limit",
        ),
        (
            ("shaders", "kernels", 1, "limits", "thread_execution_width"),
            lambda v: 0,
            "outside what the API can report",
        ),
        (("planned_work", "calls"), lambda v: v + 1, "planned work"),
        (("run", "elapsed_ns"), lambda v: 1, "run time does not cover"),
        (("run", "timed_ns"), lambda v: v + 1, "run time does not cover"),
        (("summary", "rgb_to_gray_metal_mismatched_pixels"), lambda v: 0, "summary"),
        (("summary", "contraction_fused"), lambda v: False, "summary"),
        (("config", "seed"), lambda v: 74, "config digest"),
        (("config_sha256",), lambda v: "f" * 64, "config digest"),
    ],
)
def test_resealed_tampering_is_caught_by_recomputation(path, change, message):
    with pytest.raises(KernelMetalVerificationError, match=message):
        verify_report(tampered(path, change))


def test_a_new_seed_with_a_matching_config_digest_no_longer_rebuilds():
    report = fresh()
    report["config"]["seed"] = 74
    report["config_sha256"] = config_from_mapping(report["config"]).fingerprint()

    with pytest.raises(KernelMetalVerificationError, match="case 0"):
        verify_report(reseal(report))


def test_a_consistent_report_beyond_the_level_bound_is_still_rejected():
    report = fresh()
    metal = report["correctness"]["rgb_to_gray"][1]["outputs"][0]
    reference = rgb_to_gray_reference(all_rgb_triples()).reshape(-1)
    index, _ = metal["mismatches"][0]
    level = int(reference[index]) + 2
    metal["mismatches"][0] = [index, level]
    rebuilt = reference.copy()
    for pixel, value in metal["mismatches"]:
        rebuilt[pixel] = value
    metal["buffer_sha256"] = array_sha256(rebuilt)
    metal["max_abs_difference_levels"] = 2

    with pytest.raises(KernelMetalVerificationError, match="exceeds the 1-level bound"):
        verify_report(reseal(report))
    assert numpy.count_nonzero(rebuilt != reference) == 21


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: [],
        lambda r: {},
        lambda r: {**r, "correctness": []},
        lambda r: {**r, "benchmark": {"vector_add": "x"}},
        lambda r: {**r, "shaders": {"kernels": [1, 2]}},
        lambda r: {**r, "run": None},
    ],
)
def test_malformed_reports_fail_as_verification_errors(mutate):
    report = mutate(fresh())
    if isinstance(report, dict) and report:
        report = reseal(report)

    with pytest.raises(KernelMetalVerificationError):
        verify_report(report)
    with pytest.raises(TypeError, match="KernelMetalConfig"):
        verify_report(fresh(), config=make_config_mapping())


def test_verification_runs_where_torch_cannot_be_imported(tmp_path):
    path = tmp_path / "recorded.json"
    path.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")
    script = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.kernel_metal_verify import verify_report\n"
        "report = json.loads(open(sys.argv[1], encoding='ascii').read())\n"
        "print(verify_report(report)['measured'], sys.modules['torch'] is None)\n"
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
    assert completed.stdout.split() == ["True", "True"]
