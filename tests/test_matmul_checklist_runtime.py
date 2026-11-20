import functools
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from ai_journey.kernel_metal_reference import FLOAT32_SENTINEL_BITS
from ai_journey.matmul_checklist_runtime import (
    MatmulChecklistRuntime,
    MatmulChecklistUnavailableError,
    checked_launch,
)
from ai_journey.matmul_checklist_source import MatmulChecklistError
from ai_journey.matmul_checklist_variants import (
    LIBRARY_VARIANTS,
    VARIANTS,
    checklist_library,
)
from ai_journey.matmul_metal_benchmark import agreement_bound
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    error_bound,
    float64_reference,
    fma_replay,
    matmul_inputs,
    max_relative_error,
)
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.matmul_tiled_variants import FEWER_THREADS, OTHER_THREADGROUP_BYTES
from test_kernel_metal_runtime import darwin, fake_torch, importer, requires_metal

ROOT = Path(__file__).resolve().parents[1]
NAIVE_GROUP = (32, 8)
# 17 x 33 x 9 stages on the scalar path; 65 x 68 x 40 runs the vector path,
# one interior group of the 64 x 64 kernels, edge groups and a short phase.
SCALAR = matmul_shape(17, 33, 9)
VECTOR = matmul_shape(65, 68, 40)


@functools.cache
def checklist_runtime():
    """Compile both libraries once per test process; call only under requires_metal."""

    return MatmulChecklistRuntime()


def fake_pipelines(threads=None, memory=None):
    """A torch stand-in whose libraries report chosen pipeline limits."""

    threads, memory = threads or {}, memory or {}

    def compile_shader(source):
        pipelines = {}
        for name in re.findall(r"kernel void (\w+)\(", source):
            pipelines[name] = SimpleNamespace(
                max_threads_per_threadgroup=threads.get(name, 1024),
                thread_execution_width=32,
                static_thread_group_memory_length=memory.get(
                    name, VARIANTS[name].threadgroup_bytes
                ),
            )
        return SimpleNamespace(**pipelines)

    torch = fake_torch()
    torch.mps = SimpleNamespace(compile_shader=compile_shader)
    torch.get_float32_matmul_precision = lambda: "highest"
    return torch


def fake_runtime(**limits):
    return MatmulChecklistRuntime(
        system=darwin, import_module=importer(fake_pipelines(**limits)), environ={}
    )


def launch(runtime, name, shape=SCALAR):
    return checked_launch(runtime.limits, name, shape, NAIVE_GROUP)


@pytest.mark.parametrize(
    ("module", "reason"),
    [
        (ImportError("no torch"), "torch_not_importable"),
        (fake_torch(built=False), "pytorch_without_mps"),
        (fake_torch(available=False), "mps_unavailable"),
        (fake_torch(compile_shader=False), "compile_shader_unavailable"),
    ],
)
def test_a_host_without_the_requirements_is_named_before_compiling(module, reason):
    with pytest.raises(MatmulChecklistUnavailableError, match=reason):
        MatmulChecklistRuntime(system=darwin, import_module=importer(module))
    with pytest.raises(MatmulChecklistUnavailableError, match="unsupported_operating"):
        MatmulChecklistRuntime(
            system=lambda: "Linux",
            import_module=importer(AssertionError("torch must not be imported")),
        )


@pytest.mark.parametrize(
    ("marker", "label"),
    [
        ("kernel void matmul_x_rows(", "the Day 74 naive source"),
        ("kernel void matmul_s7_interior(", "the checklist library"),
    ],
)
def test_a_rejected_library_is_named_as_an_unavailable_host(marker, label):
    torch = fake_pipelines()
    compile_shader = torch.mps.compile_shader

    def refuse(source):
        if marker in source:
            raise RuntimeError("program_source:9: error: unknown type name")
        return compile_shader(source)

    torch.mps = SimpleNamespace(compile_shader=refuse)

    with pytest.raises(MatmulChecklistUnavailableError, match=f"rejected {label}"):
        MatmulChecklistRuntime(system=darwin, import_module=importer(torch))


def test_every_pipeline_reports_its_limits_and_a_refusal_is_data():
    runtime = fake_runtime(
        threads={"matmul_s6_group_32x32": 896},
        memory={"matmul_s5_pad4": 9488},
    )

    assert list(runtime.limits) == list(VARIANTS)
    assert list(runtime.admissions) == list(LIBRARY_VARIANTS)
    reasons = {name: entry.reason for name, entry in runtime.admissions.items()}
    assert reasons == {
        **dict.fromkeys(LIBRARY_VARIANTS),
        "matmul_s5_pad4": OTHER_THREADGROUP_BYTES,
        "matmul_s6_group_32x32": FEWER_THREADS,
    }
    assert runtime.library_source == checklist_library()
    assert runtime.float32_matmul_precision == "highest"
    assert runtime.mps_switches == {
        "PYTORCH_MPS_FAST_MATH": None,
        "PYTORCH_MPS_PREFER_METAL": None,
    }


def test_a_refused_kernel_or_a_foreign_launch_never_runs():
    runtime = fake_runtime(threads={"matmul_s6_group_32x32": 512})
    left, right = matmul_inputs(77, CORRECTNESS_STREAM, 1, SCALAR)
    other = matmul_shape(21, 33, 9)

    with pytest.raises(MatmulChecklistError, match="refused by its pipeline"):
        runtime.run(
            "matmul_s6_group_32x32", left, right, launch(runtime, "matmul_s5_pad4")
        )
    for name, wrong in (
        ("matmul_s4_vec4", launch(runtime, "matmul_s3_block_4x4", other)),
        ("matmul_s3_block_4x4", launch(runtime, "matmul_s2_coarse_4x1")),
        ("matmul_x_rows", launch(runtime, "matmul_x_columns")),
    ):
        with pytest.raises(MatmulChecklistError, match="its own launch"):
            runtime.run(name, left, right, wrong)
    with pytest.raises(MatmulChecklistError, match="not a registered variant"):
        runtime.run("matmul_tiled_16", left, right, launch(runtime, "matmul_tiled_32"))
    with pytest.raises(MatmulChecklistError, match="not a registered variant"):
        launch(runtime, "torch_matmul")
    with pytest.raises(MatmulChecklistError, match="starts with the S0 kernel"):
        runtime.workload(
            left, right, {"matmul_x_rows": launch(runtime, "matmul_x_rows")}
        )


def test_importing_the_runtime_without_torch_never_loads_it():
    script = (
        "import sys; sys.modules['torch'] = None\n"
        "from ai_journey.matmul_checklist_runtime import MatmulChecklistRuntime\n"
        "try:\n"
        "    MatmulChecklistRuntime()\n"
        "except Exception as error:\n"
        "    print(type(error).__name__, error)\n"
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
    reason = (
        "torch_not_importable"
        if sys.platform == "darwin"
        else "unsupported_operating_system"
    )
    assert completed.stdout.startswith("MatmulChecklistUnavailableError ")
    assert f"({reason})" in completed.stdout


@requires_metal
def test_both_libraries_compile_and_each_kernel_gets_a_verdict_from_its_pipeline():
    runtime = checklist_runtime()

    assert set(runtime.limits) == set(VARIANTS)
    for name, admission in runtime.admissions.items():
        limits = runtime.limits[name]
        assert (
            limits.static_threadgroup_memory_bytes == VARIANTS[name].threadgroup_bytes
        )
        assert admission.admitted == (
            limits.max_threads_per_threadgroup >= admission.required_threads
        )
    for name in ("matmul_x_columns", "matmul_x_rows"):
        assert runtime.limits[name].static_threadgroup_memory_bytes == 0


@requires_metal
@pytest.mark.parametrize("shape", [SCALAR, VECTOR], ids=["scalar", "vector"])
def test_each_admitted_kernel_keeps_its_guard_tail_and_the_bound(shape):
    runtime = checklist_runtime()
    left, right = matmul_inputs(77, CORRECTNESS_STREAM, 2, shape)
    reference, magnitude = float64_reference(left, right)
    replay = fma_replay(left, right).reshape(-1).view(numpy.uint32)
    elements = shape.m * shape.n

    for name in VARIANTS:
        if name in runtime.admissions and not runtime.admissions[name].admitted:
            continue
        buffer = runtime.run(name, left, right, launch(runtime, name, shape))
        tail = buffer[elements:].view(numpy.uint32)
        assert tail.size > 0 and set(tail.tolist()) == {FLOAT32_SENTINEL_BITS}, name
        output = buffer[:elements].reshape(shape.m, shape.n)
        assert max_relative_error(output, reference, magnitude) <= error_bound(shape.k)
        # Observed on the development host's GPU; a run records it per case.
        assert numpy.array_equal(buffer[:elements].view(numpy.uint32), replay), name


@requires_metal
def test_the_workload_checks_every_side_against_the_s0_output():
    runtime = checklist_runtime()
    left, right = matmul_inputs(77, 1, 0, VECTOR)
    names = [
        name
        for name in VARIANTS
        if name not in runtime.admissions or runtime.admissions[name].admitted
    ]
    launches = {name: launch(runtime, name, VECTOR) for name in names}

    workload = runtime.workload(left, right, launches)
    for call in workload.calls:
        call()
    checks = workload.check()

    assert list(workload.sides) == [*names, TORCH_MATMUL]
    assert list(checks) == [*names[1:], TORCH_MATMUL]
    for name, result in checks.items():
        assert 0.0 <= result.disagreement <= agreement_bound(VECTOR.k), name
        if name != TORCH_MATMUL:
            assert result.same_bits_as_naive is True, name
