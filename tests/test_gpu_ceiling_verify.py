import json
import os
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

from ai_journey.gpu_ceiling_config import config_from_mapping
from ai_journey.gpu_ceiling_reference import fma_chain_values, fma_start_values
from ai_journey.gpu_ceiling_report import (
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    REPORT_DIGEST_FIELD,
)
from ai_journey.gpu_ceiling_source import GpuCeilingError
from ai_journey.gpu_ceiling_verify import (
    GpuCeilingVerificationError,
    _float32,
    _fused_significands,
    _start_significands,
    verify_report,
)
from ai_journey.memory_access_report import seal_report
from test_gpu_ceiling_config import changed, make_config_mapping
from test_gpu_ceiling_report import CONFIG, stand_in_report

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = stand_in_report()


def fresh(report=SYNTHETIC):
    return json.loads(json.dumps(report))


def reseal(report):
    unsigned = {
        key: value for key, value in report.items() if key != REPORT_DIGEST_FIELD
    }
    return seal_report(unsigned)


def tampered(path, change, report=SYNTHETIC):
    copy = fresh(report)
    target = copy
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = change(target[path[-1]])
    return reseal(copy)


def relabeled_as_measured(report=SYNTHETIC):
    copy = fresh(report)
    copy.update(
        measured=True,
        backend="torch.mps.compile_shader",
        timer="time.perf_counter_ns",
        measurement_scope=MEASUREMENT_SCOPE,
        label=MEASUREMENT_LABEL,
        evidence_boundary=dict(MEASUREMENT_BOUNDARY),
    )
    copy["summary"]["label"] = MEASUREMENT_LABEL
    copy["environment"]["platform_system"] = "Darwin"
    return copy


def test_a_synthetic_report_verifies_and_binds_to_its_config():
    summary = verify_report(fresh(), config=CONFIG)

    assert summary["kind"] == "gpu_ceiling_measurement"
    assert summary["report_sha256"] == SYNTHETIC[REPORT_DIGEST_FIELD]
    assert (summary["measured"], summary["scope"]) == (
        False,
        "synthetic_backend_or_clock",
    )
    assert summary["correctness_cases"] == 10
    assert (summary["fma_points"], summary["stream_points"]) == (6, 4)
    assert summary["timed_samples"] == 48
    assert summary["load_average_before"] == [3.5, 4.25, 5.0]
    other = config_from_mapping(changed(("seed",), 75))
    with pytest.raises(GpuCeilingVerificationError, match="supplied config"):
        verify_report(fresh(), config=other)
    with pytest.raises(TypeError, match="GpuCeilingConfig"):
        verify_report(fresh(), config=make_config_mapping())


def test_consistent_labels_verify_as_measured_but_a_digest_proves_no_timing():
    # The seal shows internal consistency only, as the guide says: relabeled
    # stand-in timings that obey every rule still verify.
    relabeled = reseal(relabeled_as_measured())

    assert verify_report(relabeled)["measured"] is True
    linux = relabeled_as_measured()
    linux["environment"]["platform_system"] = "Linux"
    with pytest.raises(GpuCeilingVerificationError, match="must come from macOS"):
        verify_report(reseal(linux))
    half = fresh()
    half["measured"] = True
    with pytest.raises(GpuCeilingVerificationError, match="only the host timer"):
        verify_report(reseal(half))


def test_an_edit_without_a_new_seal_breaks_the_digest():
    report = fresh()
    report["benchmark"]["fma"][0]["samples_ns"][0] += 1

    with pytest.raises(GpuCeilingVerificationError, match="digest does not match"):
        verify_report(report)


FMA_CASE = ("correctness", "fma", 0)
COPY_CASE = ("correctness", "stream", 0)
REDUCE_CASE = ("correctness", "stream", 1)
BANK_CASE = ("correctness", "bank", 0)
FMA_POINT = ("benchmark", "fma", 5)
COPY_POINT = ("benchmark", "stream", 1)
BANK_POINT = ("benchmark", "bank", 2)
ZEROS = "0" * 64


@pytest.mark.parametrize(
    ("path", "change", "message"),
    [
        ((*FMA_CASE, "buffer_sha256"), lambda v: ZEROS, "fma_chains_1 case"),
        ((*FMA_CASE, "start_sha256"), lambda v: ZEROS, "fma_chains_1 case"),
        ((*FMA_CASE, "guard_words"), lambda v: v + 1, "fma_chains_1 case"),
        ((*COPY_CASE, "source_sha256"), lambda v: ZEROS, "stream_copy case"),
        ((*REDUCE_CASE, "expected_sha256"), lambda v: ZEROS, "stream_read_reduce case"),
        (
            (*BANK_CASE, "buffer_sha256"),
            lambda v: ZEROS,
            "bank_rows_32 case at stride 1",
        ),
        (("correctness", "fma_coefficients"), lambda v: [1.0, 0.0], "references"),
        (("correctness", "bank"), lambda v: v[:-1], "both layouts"),
        ((*FMA_POINT, "samples_ns"), lambda v: [*v[:-1], v[-1] + 7], "recompute"),
        ((*FMA_POINT, "samples_ns"), lambda v: [True, *v[1:]], "positive integers"),
        ((*FMA_POINT, "samples_ns"), lambda v: v[:-1], "positive integers"),
        ((*FMA_POINT, "output_sha256"), lambda v: ZEROS, "fma_chains_4 point"),
        ((*FMA_POINT, "median_gflops_per_second"), lambda v: v * 2, "recompute"),
        ((*COPY_POINT, "calls_per_sample"), lambda v: v + 1, "stream_copy point"),
        ((*COPY_POINT, "output_sha256"), lambda v: ZEROS, "stream_copy point"),
        ((*BANK_POINT, "median_ps_per_access"), lambda v: v + 1, "bank_rows_32 point"),
        ((*BANK_POINT, "launch", "padding_threads"), lambda v: 1, "bank_rows_32"),
        (("ceilings", "fma", "gflops_per_second"), lambda v: v * 10, "ceilings"),
        (("ceilings", "fma", "scales_with_work"), lambda v: not v, "ceilings"),
        (
            ("ceilings", "bank", "finding"),
            lambda v: "no_clear_conflict_pattern",
            "ceilings",
        ),
        (("ceilings", "definition"), lambda v: "the vendor peak", "ceilings"),
        (("ceilings", "ridge_point_flops_per_byte"), lambda v: v + 1, "ceilings"),
        (("summary", "bank_finding"), lambda v: "other", "summary"),
        (("summary", "load_average_after"), lambda v: [0.0, 0.0, 0.0], "summary"),
        (("host_load", "before"), lambda v: [-1.0, 1.0, 1.0], "host_load"),
        (("host_load", "after"), lambda v: [1, 1, 1], "host_load"),
        (("environment", "torch_version"), lambda v: "2.15 local", "token"),
        (("shaders", "source"), lambda v: v + "\n", "Metal source"),
        (
            ("shaders", "kernels", 9, "limits", "static_threadgroup_memory_bytes"),
            lambda v: 0,
            "hold its tile",
        ),
        (
            ("shaders", "kernels", 0, "limits", "max_threads_per_threadgroup"),
            lambda v: 0,
            "outside",
        ),
        (("planned_work", "flops"), lambda v: v + 1, "planned work"),
        (("run", "elapsed_ns"), lambda v: 1, "run time"),
        (("run", "timed_ns"), lambda v: v + 1, "run time"),
        (("evidence_boundary", "vendor_peak_used"), lambda v: True, "weakened"),
        (("label",), lambda v: MEASUREMENT_LABEL, "weakened"),
        (("config", "seed"), lambda v: 75, "config digest"),
        (("percentile_method",), lambda v: "mean", "method fields"),
        (("schema_version",), lambda v: 2, "schema"),
        (("kind",), lambda v: "kernel_metal_correctness_and_benchmark", "kind"),
    ],
)
def test_resealed_edits_are_named_by_the_check_that_fails(path, change, message):
    with pytest.raises(GpuCeilingVerificationError, match=message):
        verify_report(tampered(path, change))


def test_a_changed_config_with_a_matching_digest_fails_the_rebuild():
    copy = fresh()
    copy["config"]["seed"] = 75
    copy["config_sha256"] = config_from_mapping(copy["config"]).fingerprint()

    with pytest.raises(GpuCeilingVerificationError, match="planned work|case"):
        verify_report(reseal(copy))


def test_unknown_or_missing_fields_and_bad_roots_are_refused():
    extra = fresh()
    extra["note"] = "x"
    missing = fresh()
    del missing["host_load"]

    for report in (reseal(extra), reseal(missing)):
        with pytest.raises(GpuCeilingVerificationError, match="report fields"):
            verify_report(report)
    with pytest.raises(GpuCeilingVerificationError, match="JSON object"):
        verify_report([])
    assert issubclass(GpuCeilingVerificationError, GpuCeilingError)


def test_the_integer_rebuild_agrees_with_the_float64_reference():
    start = fma_start_values(74)
    floats = fma_chain_values(start, (1, 64, 8192))
    integers = _fused_significands(_start_significands(74), [1, 64, 8192])

    assert numpy.array_equal(_float32(_start_significands(74)), start)
    for steps, table in zip((1, 64, 8192), floats):
        assert numpy.array_equal(_float32(integers[steps]), table), steps


def test_verify_never_imports_torch(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(SYNTHETIC), encoding="ascii")
    script = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.gpu_ceiling_verify import verify_report\n"
        "report = json.loads(open(sys.argv[1], encoding='ascii').read())\n"
        "print(verify_report(report)['report_sha256'])\n"
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
    assert completed.stdout.strip() == SYNTHETIC[REPORT_DIGEST_FIELD]
