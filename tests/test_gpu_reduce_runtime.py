import functools
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from ai_journey.gpu_reduce_bound import (
    VENDOR_DEPTH,
    ReductionCheck,
    benchmark_values,
    case_values,
    exact_sums,
    partial_count,
    summation_depth,
)
from ai_journey.gpu_reduce_runtime import (
    FEWER_THREADS,
    INPUT_GUARD,
    OTHER_SIMD_WIDTH,
    OTHER_THREADGROUP_BYTES,
    OUTPUT_SLOTS,
    PARTIAL_GUARD,
    GpuReduceRuntime,
    GpuReduceUnavailableError,
    checked_launches,
)
from ai_journey.gpu_reduce_source import (
    COMBINE,
    KERNELS,
    R0,
    R2,
    R3,
    R4,
    STEP_KERNELS,
    GpuReduceError,
    library_source,
)
from ai_journey.kernel_metal_geometry import CheckedLaunch
from ai_journey.kernel_metal_reference import FLOAT32_SENTINEL_BITS
from test_kernel_metal_runtime import darwin, fake_torch, importer, requires_metal

ROOT = Path(__file__).resolve().parents[1]
SIDES = [*STEP_KERNELS, "torch_sum"]


@functools.cache
def reduce_runtime():
    """Compile the library once per test process; call only under requires_metal."""

    return GpuReduceRuntime()


def fake_pipelines(threads=None, width=None, memory=None):
    """A torch stand-in whose library reports chosen pipeline limits."""

    threads, width, memory = threads or {}, width or {}, memory or {}

    def compile_shader(source):
        pipelines = {}
        for name in re.findall(r"kernel void (\w+)\(", source):
            pipelines[name] = SimpleNamespace(
                max_threads_per_threadgroup=threads.get(name, 1024),
                thread_execution_width=width.get(name, 32),
                static_thread_group_memory_length=memory.get(
                    name, KERNELS[name].threadgroup_bytes
                ),
            )
        return SimpleNamespace(**pipelines)

    torch = fake_torch()
    torch.mps = SimpleNamespace(compile_shader=compile_shader)
    return torch


def fake_runtime(environ=None, **limits):
    return GpuReduceRuntime(
        system=darwin,
        import_module=importer(fake_pipelines(**limits)),
        environ=environ or {},
    )


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
    with pytest.raises(GpuReduceUnavailableError, match=reason):
        GpuReduceRuntime(system=darwin, import_module=importer(module))
    with pytest.raises(GpuReduceUnavailableError, match="unsupported_operating"):
        GpuReduceRuntime(
            system=lambda: "Linux",
            import_module=importer(AssertionError("torch must not be imported")),
        )


def test_a_rejected_library_is_named_as_an_unavailable_host():
    torch = fake_pipelines()

    def refuse(source):
        raise RuntimeError("program_source:12: error: use of undeclared identifier")

    torch.mps = SimpleNamespace(compile_shader=refuse)
    with pytest.raises(GpuReduceUnavailableError, match="rejected the reduction"):
        GpuReduceRuntime(system=darwin, import_module=importer(torch))


def test_every_pipeline_reports_its_limits_and_admits_its_kernel():
    runtime = fake_runtime(environ={"PYTORCH_MPS_FAST_MATH": "1"})

    assert list(runtime.limits) == list(KERNELS)
    assert runtime.library_source == library_source()
    assert all(entry.admitted for entry in runtime.admissions.values())
    assert runtime.admissions[R4].to_mapping() == {
        "name": R4,
        "admitted": True,
        "reason": None,
        "required_threads": 256,
        "allowed_threads": 1024,
        "required_simd_width": 32,
        "simd_width": 32,
        "declared_threadgroup_bytes": 32,
        "static_threadgroup_bytes": 32,
    }
    assert runtime.mps_switches == {
        "PYTORCH_MPS_FAST_MATH": "1",
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    assert runtime.torch_version == "2.15.0"


@pytest.mark.parametrize(
    ("limits", "name", "reason"),
    [
        ({"threads": {R0: 128}}, R0, FEWER_THREADS),
        ({"threads": {COMBINE: 512}}, COMBINE, FEWER_THREADS),
        ({"width": {R3: 64}}, R3, OTHER_SIMD_WIDTH),
        ({"memory": {COMBINE: 4112}}, COMBINE, OTHER_THREADGROUP_BYTES),
    ],
)
def test_one_refused_kernel_stops_the_runtime_before_any_case(limits, name, reason):
    with pytest.raises(GpuReduceError, match=f"{name} was refused .*: {reason}"):
        fake_runtime(**limits)


def test_a_simd_width_matters_only_to_the_kernels_that_use_simd_groups():
    runtime = fake_runtime(width={R0: 64, COMBINE: 16})

    assert runtime.admissions[R0].admitted and runtime.admissions[COMBINE].admitted


def test_each_launch_runs_only_on_its_own_pipeline():
    runtime = fake_runtime()
    first, second = checked_launches(runtime.limits, R0, 1000003)

    assert first.geometry.groups == (3907,) and first.geometry.group_size == (256,)
    assert second.geometry.groups == (1,) and second.geometry.group_size == (1024,)
    assert partial_count(R0, 1000003) == 3907
    with pytest.raises(GpuReduceError, match="its own pipeline's launch"):
        runtime._kernel(R4, CheckedLaunch(first.geometry, runtime.limits[R0]))
    with pytest.raises(GpuReduceError, match="not a step kernel"):
        runtime.run(COMBINE, numpy.ones(4, dtype=numpy.float32))
    with pytest.raises(GpuReduceError, match="1D float32 array"):
        runtime.run(R0, numpy.ones(4))


def test_importing_the_runtime_without_torch_never_loads_it():
    script = (
        "import sys; sys.modules['torch'] = None\n"
        "from ai_journey.gpu_reduce_runtime import GpuReduceRuntime\n"
        "try:\n"
        "    GpuReduceRuntime()\n"
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
    assert completed.stdout.startswith("GpuReduceUnavailableError ")
    assert f"({reason})" in completed.stdout


@requires_metal
def test_the_library_compiles_and_every_pipeline_admits_its_kernel():
    runtime = reduce_runtime()

    for name, admission in runtime.admissions.items():
        limits = runtime.limits[name]
        assert admission.admitted, admission.to_mapping()
        assert limits.static_threadgroup_memory_bytes == KERNELS[name].threadgroup_bytes
        assert limits.thread_execution_width == 32


@requires_metal
@pytest.mark.parametrize(
    ("pattern", "n"),
    [("normal", 1), ("normal", 33), ("mixed", 300007), ("ones", 1025)],
)
def test_each_kernel_keeps_its_guard_tails_and_its_bound(pattern, n):
    runtime = reduce_runtime()
    values = case_values(78, 5, pattern, n)
    sums = exact_sums(values)
    sentinel = numpy.uint32(FLOAT32_SENTINEL_BITS)

    outputs = {}
    for name in STEP_KERNELS:
        buffers = runtime.run(name, values)
        groups = partial_count(name, n)
        assert buffers.partials.size == groups + PARTIAL_GUARD
        assert numpy.isfinite(buffers.partials[:groups]).all(), name
        assert (buffers.partials[groups:].view(numpy.uint32) == sentinel).all()
        assert buffers.output.size == OUTPUT_SLOTS
        assert (buffers.output[1:].view(numpy.uint32) == sentinel).all()
        output = float(buffers.output[0])
        assert ReductionCheck(output, summation_depth(name, n), sums).within, name
        outputs[name] = output
    # R3 adds the same pairs in the same order as R2; observed on this host.
    assert outputs[R3] == outputs[R2]
    assert INPUT_GUARD == 1024


@requires_metal
def test_the_workload_binds_every_side_to_one_device_buffer():
    runtime = reduce_runtime()
    values = benchmark_values(78, 1 << 20)
    source = runtime.upload(values)
    n = 300007
    sums = exact_sums(values[:n])

    workload = runtime.workload(source, n)
    for call in workload.calls:
        call()
    results = workload.check()

    assert list(workload.sides) == SIDES and list(results) == SIDES
    for name in STEP_KERNELS:
        check = ReductionCheck(results[name].output, summation_depth(name, n), sums)
        assert check.within and results[name].same_bits_on_repeat, name
    torch_sum = ReductionCheck(results["torch_sum"].output, VENDOR_DEPTH, sums)
    assert torch_sum.within
    with pytest.raises(GpuReduceError, match="at least 1048577 values"):
        runtime.workload(source, (1 << 20) + 1)
