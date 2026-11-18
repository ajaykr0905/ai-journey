import functools
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from ai_journey.kernel_metal_geometry import (
    CheckedLaunch,
    KernelMetalError,
    PipelineLimits,
    vector_launch,
)
from ai_journey.kernel_metal_reference import (
    FLOAT32_SENTINEL_BITS,
    vector_add_inputs,
    vector_add_reference,
)
from ai_journey.kernel_metal_runtime import KernelMetalUnavailableError
from ai_journey.kernel_metal_source import VECTOR_ADD
from ai_journey.metal_launch_sweep_layout import observe_simd_layout
from ai_journey.metal_launch_sweep_runtime import (
    PIPELINES,
    MetalLaunchSweepRuntime,
    pipeline_limits,
)
from ai_journey.metal_launch_sweep_shapes import (
    ShapeAdmission,
    checked_launch,
    matmul_launch,
)
from ai_journey.metal_launch_sweep_source import MATMUL, SWEEP_SOURCE
from test_kernel_metal_runtime import darwin, fake_torch, importer, requires_metal

ROOT = Path(__file__).resolve().parents[1]
SWEPT = ((8, 8), (16, 16), (32, 32), (32, 8), (8, 32), (64, 4), (4, 64))


@functools.cache
def sweep_runtime():
    """Compile both libraries once per test process; call only under requires_metal."""

    return MetalLaunchSweepRuntime()


def compiling_torch(*, reject_sweep):
    """A stand-in torch whose compiler returns pipelines that report fixed limits."""

    kernel = SimpleNamespace(
        max_threads_per_threadgroup=896,
        thread_execution_width=32,
        static_thread_group_memory_length=0,
    )

    def compile_shader(source):
        if reject_sweep and source == SWEEP_SOURCE:
            raise RuntimeError("program_source:12: error: unknown type name")
        return SimpleNamespace(
            vector_add=kernel,
            rgb_to_gray=kernel,
            matmul_naive=kernel,
            simd_layout_probe=kernel,
        )

    torch = fake_torch()
    torch.mps = SimpleNamespace(compile_shader=compile_shader)
    return torch


@pytest.mark.parametrize(
    ("system", "module", "reason"),
    [
        (lambda: "Linux", AssertionError("torch must not be imported"), None),
        (darwin, ImportError("no torch"), "torch_not_importable"),
        (darwin, fake_torch(built=False), "pytorch_without_mps"),
        (darwin, fake_torch(available=False), "mps_unavailable"),
    ],
)
def test_a_host_without_mps_stops_with_the_day_73_reason(system, module, reason):
    expected = reason or "unsupported_operating_system"

    with pytest.raises(KernelMetalUnavailableError, match=f"\\({expected}\\)"):
        MetalLaunchSweepRuntime(system=system, import_module=importer(module))


def test_the_runtime_reads_every_pipeline_and_states_its_simd_rule():
    runtime = MetalLaunchSweepRuntime(
        system=darwin, import_module=importer(compiling_torch(reject_sweep=False))
    )

    assert set(PIPELINES) <= set(runtime.limits)
    assert {runtime.limits[name] for name in PIPELINES} == {PipelineLimits(896, 32, 0)}
    assert MetalLaunchSweepRuntime.requires_whole_simd_groups is False
    with pytest.raises(KernelMetalUnavailableError, match="rejected the sweep source"):
        MetalLaunchSweepRuntime(
            system=darwin, import_module=importer(compiling_torch(reject_sweep=True))
        )
    kernel = SimpleNamespace(
        max_threads_per_threadgroup=1024,
        thread_execution_width=32,
        static_thread_group_memory_length=64,
    )
    assert pipeline_limits(kernel) == PipelineLimits(1024, 32, 64)


def test_importing_the_runtime_without_torch_never_loads_it():
    script = (
        "import sys; sys.modules['torch'] = None\n"
        "from ai_journey.metal_launch_sweep_runtime import MetalLaunchSweepRuntime\n"
        "try:\n"
        "    MetalLaunchSweepRuntime()\n"
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
    on_macos = sys.platform == "darwin"
    reason = "torch_not_importable" if on_macos else "unsupported_operating_system"
    assert completed.stdout.startswith("KernelMetalUnavailableError ")
    assert f"({reason})" in completed.stdout


@requires_metal
def test_the_compiled_pipelines_report_usable_limits():
    runtime = sweep_runtime()

    for name in PIPELINES:
        limits = runtime.limits[name]
        assert isinstance(limits, PipelineLimits)
        assert limits.max_threads_per_threadgroup >= 256, name
        assert limits.static_threadgroup_memory_bytes == 0, name


@requires_metal
def test_the_gpu_splits_every_swept_shape_by_the_linear_layout():
    runtime = sweep_runtime()

    for shape in SWEPT:
        admission = ShapeAdmission(shape, runtime.limits[MATMUL], False)
        assert observe_simd_layout(runtime, admission).group_size == shape
    for size in (8, 16, 48, 1024):
        admission = ShapeAdmission((size,), runtime.limits[VECTOR_ADD], False)
        layout = observe_simd_layout(runtime, admission)
        assert layout.footprint.rows == 1


def float64_bound(a, b):
    """Return |reference| and the float32 rounding bound of each element of A B."""

    inner = a.shape[1]
    gamma = inner * 2.0**-24 / (1 - inner * 2.0**-24)
    left, right = a.astype(numpy.float64), b.astype(numpy.float64)
    return left @ right, gamma * (numpy.abs(left) @ numpy.abs(right)) + 1e-300


@requires_metal
def test_the_matmul_guards_its_padding_and_is_the_same_for_every_shape():
    runtime = sweep_runtime()
    generator = numpy.random.default_rng(74)
    m, n, k = 37, 53, 29
    a = generator.standard_normal((m, k), dtype=numpy.float32)
    b = generator.standard_normal((k, n), dtype=numpy.float32)
    reference, bound = float64_bound(a, b)

    outputs = []
    for shape in SWEPT:
        admission = ShapeAdmission(shape, runtime.limits[MATMUL], False)
        launch = checked_launch(admission, matmul_launch(m, n, shape))
        buffer = runtime.matmul(a, b, launch)
        assert buffer.shape == (m * n + launch.geometry.padding_threads,)
        tail = buffer[m * n :].view(numpy.uint32)
        assert set(tail.tolist()) <= {FLOAT32_SENTINEL_BITS}, shape
        outputs.append(buffer[: m * n].reshape(m, n))
    for output in outputs:
        assert numpy.array_equal(output.view(numpy.uint32), outputs[0].view("u4"))
    assert (numpy.abs(outputs[0] - reference) <= bound).all()


@requires_metal
def test_workloads_bind_one_call_per_shape_and_compare_their_outputs():
    runtime = sweep_runtime()
    generator = numpy.random.default_rng(75)
    a = generator.standard_normal((64, 32), dtype=numpy.float32)
    b = generator.standard_normal((32, 96), dtype=numpy.float32)
    shapes = ((16, 16), (64, 4), (4, 64))
    launches = tuple(
        checked_launch(
            ShapeAdmission(shape, runtime.limits[MATMUL], False),
            matmul_launch(64, 96, shape),
        )
        for shape in shapes
    )

    matmul = runtime.matmul_workload(a, b, launches)
    for call in matmul.calls:
        call()
    assert matmul.outputs_agree()
    single = runtime.matmul(a, b, launches[0])[: 64 * 96]
    assert numpy.array_equal(matmul.output().view("u4"), single.view("u4"))

    first, second = vector_add_inputs(74, 0, 1000)
    vector_launches = tuple(
        CheckedLaunch(vector_launch(1000, size), runtime.limits[VECTOR_ADD])
        for size in (8, 48, 256)
    )
    vector = runtime.vector_workload(first, second, vector_launches)
    for call in vector.calls:
        call()
    assert vector.outputs_agree()
    expected = vector_add_reference(first, second).view(numpy.uint32)
    assert numpy.array_equal(vector.output().view(numpy.uint32), expected)


@requires_metal
def test_launches_must_fit_the_kernel_pipeline_and_the_extent():
    runtime = sweep_runtime()
    a = numpy.ones((8, 4), dtype=numpy.float32)
    b = numpy.ones((4, 8), dtype=numpy.float32)
    other = CheckedLaunch(matmul_launch(8, 8, (8, 8)), PipelineLimits(512, 32, 0))
    wrong_extent = checked_launch(
        ShapeAdmission((8, 8), runtime.limits[MATMUL], False),
        matmul_launch(8, 16, (8, 8)),
    )

    for launch in (other, wrong_extent):
        with pytest.raises(KernelMetalError, match="checked against its own pipeline"):
            runtime.matmul(a, b, launch)
    with pytest.raises(KernelMetalError, match="conformable"):
        runtime.matmul(a, a, wrong_extent)
    first, second = vector_add_inputs(74, 1, 64)
    mixed = (
        CheckedLaunch(vector_launch(64, 32), runtime.limits[VECTOR_ADD]),
        CheckedLaunch(vector_launch(65, 32), runtime.limits[VECTOR_ADD]),
    )
    with pytest.raises(KernelMetalError, match="one length"):
        runtime.vector_workload(first, second, mixed)
