import dataclasses
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

from ai_journey.gpu_reduce_ceiling import load_ceiling
from ai_journey.gpu_reduce_config import config_from_mapping
from ai_journey.gpu_reduce_report import MEASUREMENT_BOUNDARY, measure_report
from ai_journey.gpu_reduce_verify import GpuReduceVerificationError, verify_report
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import seal_report
from test_gpu_reduce_config import make_config_mapping
from test_gpu_reduce_correctness import StandInBackend
from test_gpu_reduce_speedup import write_ceiling

ROOT = Path(__file__).resolve().parents[1]
LOAD = (1.5, 2.25, 3.0)
R0 = "reduce_r0_interleaved"


@pytest.fixture(scope="module")
def ceiling(tmp_path_factory):
    return load_ceiling(write_ceiling(tmp_path_factory.mktemp("ceiling")))


@pytest.fixture(scope="module")
def report(ceiling):
    return measure_report(
        config_from_mapping(make_config_mapping()),
        ceiling,
        backend=StandInBackend(flaky={"reduce_r5_grid_stride"}),
        clock=itertools.count(0, 1_000).__next__,
        load_average=lambda: LOAD,
    )


def resealed(report, change):
    """Apply one change to a copy, then seal it again, so the digest matches."""

    copy = json.loads(json.dumps(report))
    change(copy)
    copy.pop("report_sha256")
    return seal_report(copy)


def ulp_up(value):
    return float(numpy.nextafter(numpy.float32(value), numpy.float32(numpy.inf)))


def set_path(*path, value):
    def change(copy):
        target = copy
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


def measured_look(copy):
    copy["measured"] = True
    copy["timer"] = "time.perf_counter_ns"
    copy["backend"] = "torch.mps.compile_shader"
    copy["measurement_scope"] = "apple_gpu_metal_this_host"
    copy["label"] = "host-specific Apple GPU (Metal) measurement"
    copy["evidence_boundary"] = dict(MEASUREMENT_BOUNDARY)
    copy["environment"] = {**copy["environment"], "platform_system": "Darwin"}
    copy["ceiling"]["measured"] = False
    copy["ceiling"]["label"] = "synthetic timing, not a measurement"


def test_a_stand_in_report_verifies_as_synthetic_with_its_inputs(report, ceiling):
    config = config_from_mapping(make_config_mapping())

    summary = verify_report(report, config=config, ceiling=ceiling)

    assert summary["report_sha256"] == report["report_sha256"]
    assert (summary["measured"], summary["scope"]) == (
        False,
        "synthetic_backend_or_clock",
    )
    assert summary["ceiling_checked"] is True
    assert summary["cases_with_changed_bits"]["reduce_r5_grid_stride"] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
    assert summary["exact_cases"] == [0, 4, 5]
    assert verify_report(report)["ceiling_checked"] is False


TAMPERS = [
    (set_path("measured", value=True), "only the host timer"),
    (
        set_path("label", value="host-specific Apple GPU (Metal) measurement"),
        "evidence labels or boundary were weakened",
    ),
    (
        set_path("evidence_boundary", "cuda_kernel_executed", value=True),
        "weakened",
    ),
    (set_path("config", "seed", value=79), "config digest does not match"),
    (
        set_path("plan", "steps", 1, "hypothesis", value="R1 is faster."),
        "does not seal the declared plan",
    ),
    (
        set_path("environment", "torch_version", value="2.15 beta"),
        "version or platform token",
    ),
    (
        set_path(
            "shaders",
            "kernels",
            0,
            "limits",
            "static_threadgroup_memory_bytes",
            value=2048,
        ),
        "was refused by its pipeline",
    ),
    (
        set_path("shaders", "kernels", 3, "source_sha256", value="0" * 64),
        "shader section does not recompute",
    ),
    (set_path("ceiling", "gb_per_second", value=99.0), "speedup rows"),
    (set_path("ceiling", "same_environment", value=None), "same_environment"),
    (set_path("ceiling", "kernel", value="fma_chains_8"), "verified gpu-ceilings"),
    (measured_look, "measured ceiling of its own environment"),
    (set_path("planned_work", "calls", value=1), "planned work"),
    (
        set_path("correctness", "cases", 1, "input_sha256", value="0" * 64),
        "case 1 does not rebuild from the seed",
    ),
    (
        set_path("correctness", "cases", 1, "outputs", 2, "output", value=1e6),
        "outside the bound of its order",
    ),
    (
        set_path("correctness", "cases", 4, "outputs", 0, "output", value=ulp_up(1025)),
        "misses an exact sum",
    ),
    (
        set_path("correctness", "cases", 1, "outputs", 0, "output", value=0.1),
        "finite float32 result",
    ),
    (
        set_path("correctness", "cases", 2, "outputs", 1, "depth", value=99),
        "case 2 does not rebuild",
    ),
    (
        set_path(
            "correctness", "cases", 0, "outputs", 4, "same_bits_every_run", value=1
        ),
        "must record its run bits",
    ),
    (
        set_path("correctness", "cases", 0, "same_order", 0, "same_bits", value=False),
        "case 0 does not rebuild",
    ),
    (
        set_path("benchmark", "points", 0, "sides", R0, "samples_ns", 0, value=7),
        "does not recompute from its samples",
    ),
    (
        set_path("benchmark", "points", 1, "sides", R0, "samples_ns", value=[1, 2]),
        "samples_ns must hold 3 positive integers",
    ),
    (
        set_path("benchmark", "points", 1, "checks", "torch_sum", "output", value=1e9),
        "outside the bound",
    ),
    (
        set_path("benchmark", "points", 0, "reference", value=0.0),
        "does not recompute",
    ),
    (
        set_path("speedup", "points", 0, "rows", 2, "held", value=False),
        "speedup rows",
    ),
    (
        set_path(
            "speedup", "points", 1, "rows", 1, "to_base", "verdict", value="faster"
        ),
        "speedup rows",
    ),
    (set_path("run", "elapsed_ns", value=1), "run time does not cover"),
    (set_path("summary", "exact_cases", value=[0]), "summary does not reconcile"),
    (set_path("warnings", "count", value=3), "warning counts do not reconcile"),
    (set_path("host_load", "after", value=[1.0, 2.0]), "host_load must hold"),
    (
        set_path("torch_mps_switches", "values", "PYTORCH_MPS_FAST_MATH", value="a b"),
        "MPS switches",
    ),
    (set_path("extra", value=1), "fields do not match the schema"),
]


@pytest.mark.parametrize(("change", "message"), TAMPERS)
def test_a_tampered_report_fails_even_when_sealed_again(report, change, message):
    tampered = resealed(report, change)

    with pytest.raises(GpuReduceVerificationError, match=message):
        verify_report(tampered)


def test_an_unsealed_change_and_foreign_inputs_are_named(report, ceiling):
    changed = {**report, "label": "host-specific Apple GPU (Metal) measurement"}
    with pytest.raises(GpuReduceVerificationError, match="digest does not match"):
        verify_report(changed)
    other = config_from_mapping({**make_config_mapping(), "seed": 79})
    with pytest.raises(GpuReduceVerificationError, match="supplied config"):
        verify_report(report, config=other)
    faster = dataclasses.replace(ceiling, gb_per_second=ceiling.gb_per_second * 2)
    with pytest.raises(GpuReduceVerificationError, match="supplied gpu-ceilings"):
        verify_report(report, ceiling=faster)
    with pytest.raises(TypeError, match="BandwidthCeiling"):
        verify_report(report, ceiling={})
    with pytest.raises(GpuReduceVerificationError, match="JSON object"):
        verify_report([report])


def test_the_verifier_never_imports_torch(report, tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(canonical_json_bytes(report) + b"\n")
    script = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.gpu_reduce_verify import verify_report\n"
        "summary = verify_report(json.load(open(sys.argv[1])))\n"
        "print(summary['report_sha256'], 'torch' in sys.modules and "
        "sys.modules['torch'] is not None)\n"
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
    assert completed.stdout.split() == [report["report_sha256"], "False"]
