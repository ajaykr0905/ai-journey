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
    image_launch,
    vector_launch,
)
from ai_journey.kernel_metal_reference import (
    FLOAT32_SENTINEL_BITS,
    MAX_LEVEL_DIFFERENCE,
    UINT8_SENTINEL,
    compare_levels,
    rgb_to_gray_reference,
    uniform_rgb_image,
    vector_add_inputs,
    vector_add_reference,
)
from ai_journey.kernel_metal_runtime import (
    FLOAT32_WEIGHTS,
    KernelMetalUnavailableError,
    MetalAvailability,
    MetalRuntime,
    probe_metal,
)
from ai_journey.kernel_metal_source import PROBE_FUSED_RESULT, PROBE_UNFUSED_RESULT

ROOT = Path(__file__).resolve().parents[1]
AVAILABILITY = probe_metal()
requires_metal = pytest.mark.skipif(
    not AVAILABILITY.available,
    reason="needs an Apple GPU through torch.mps.compile_shader; "
    f"this host: {AVAILABILITY.message}",
)


@functools.cache
def metal_runtime():
    """Compile the kernels once per test process; call only under requires_metal."""

    return MetalRuntime()


def fake_torch(built=True, available=True, compile_shader=True, native=True):
    return SimpleNamespace(
        __version__="2.15.0",
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_built=lambda: built, is_available=lambda: available)
        ),
        mps=SimpleNamespace(compile_shader=print) if compile_shader else None,
        _C=SimpleNamespace(_mps_compileShader=print) if native else None,
    )


def importer(module):
    def import_module(name):
        assert name == "torch"
        if isinstance(module, Exception):
            raise module
        return module

    return import_module


def darwin():
    return "Darwin"


@pytest.mark.parametrize(
    ("system", "module", "reason"),
    [
        (lambda: "Linux", AssertionError("torch must not be imported"), None),
        (darwin, ImportError("no torch"), "torch_not_importable"),
        (darwin, OSError("broken library"), "torch_not_importable"),
        (darwin, fake_torch(built=False), "pytorch_without_mps"),
        (darwin, fake_torch(available=False), "mps_unavailable"),
        (darwin, fake_torch(compile_shader=False), "compile_shader_unavailable"),
        (darwin, fake_torch(native=False), "compile_shader_unavailable"),
    ],
)
def test_the_probe_names_the_first_missing_requirement(system, module, reason):
    availability = probe_metal(system=system, import_module=importer(module))

    expected = reason or "unsupported_operating_system"
    assert (availability.available, availability.reason) == (False, expected)
    assert f"({expected})" in availability.message
    assert "needs an Apple GPU on macOS" in availability.message
    with pytest.raises(KernelMetalUnavailableError, match=expected):
        MetalRuntime(system=system, import_module=importer(module))


def test_a_complete_runtime_is_reported_available_with_its_version():
    availability = probe_metal(system=darwin, import_module=importer(fake_torch()))

    assert availability == MetalAvailability(True, None, "2.15.0")
    assert (
        availability.message == "Metal kernels can run through torch.mps.compile_shader"
    )
    with pytest.raises(KernelMetalError, match="one known reason"):
        MetalAvailability(False, "no_reason_given", None)
    with pytest.raises(KernelMetalError, match="one known reason"):
        MetalAvailability(True, "mps_unavailable", None)


def test_importing_and_probing_without_torch_never_loads_it():
    script = (
        "import sys; sys.modules['torch'] = None\n"
        "from ai_journey.kernel_metal_runtime import probe_metal\n"
        "print(probe_metal().reason)\n"
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
    expected = "torch_not_importable" if on_macos else "unsupported_operating_system"
    assert completed.stdout.strip() == expected


def test_float32_weights_are_the_reference_weights_rounded_once():
    assert [numpy.float32(weight) for weight in FLOAT32_WEIGHTS] == [
        numpy.float32(0.2989),
        numpy.float32(0.5870),
        numpy.float32(0.1140),
    ]


@requires_metal
def test_the_compiled_pipelines_report_usable_limits():
    runtime = metal_runtime()

    assert set(runtime.limits) == {"vector_add", "rgb_to_gray"}
    for limits in runtime.limits.values():
        assert isinstance(limits, PipelineLimits)
        assert limits.max_threads_per_threadgroup >= 256
    assert runtime.contraction_probe() in (PROBE_UNFUSED_RESULT, PROBE_FUSED_RESULT)
    assert runtime.torch_version == AVAILABILITY.torch_version


@requires_metal
def test_vector_add_matches_the_reference_and_leaves_the_guard_tail():
    runtime = metal_runtime()
    first, second = vector_add_inputs(73, 0, 1000)
    launch = CheckedLaunch(vector_launch(1000, 256), runtime.limits["vector_add"])
    reference = vector_add_reference(first, second).view(numpy.uint32)

    buffer = runtime.vector_add(first, second, launch)

    assert buffer.shape == (1024,)
    assert numpy.array_equal(buffer[:1000].view(numpy.uint32), reference)
    assert set(buffer[1000:].view(numpy.uint32).tolist()) == {FLOAT32_SENTINEL_BITS}
    for device in ("mps", "cpu"):
        result = runtime.torch_vector_add(first, second, device)
        assert numpy.array_equal(result.view(numpy.uint32), reference), device


@requires_metal
def test_grayscale_stays_within_the_bound_and_leaves_the_guard_tail():
    runtime = metal_runtime()
    image = uniform_rgb_image(73, 1, 17, 33)
    launch = CheckedLaunch(
        image_launch(17, 33, (16, 16)), runtime.limits["rgb_to_gray"]
    )
    reference = rgb_to_gray_reference(image)

    buffer = runtime.rgb_to_gray(image, launch)

    assert buffer.shape == (17 * 33 + 975,)
    comparison = compare_levels(buffer[: 17 * 33].reshape(17, 33), reference)
    assert comparison.max_abs_difference <= MAX_LEVEL_DIFFERENCE
    assert set(buffer[17 * 33 :].tolist()) == {UINT8_SENTINEL}
    for device in ("mps", "cpu"):
        result = runtime.torch_rgb_to_gray(image, device)
        assert numpy.array_equal(result, reference), device


@requires_metal
def test_workloads_bind_both_sides_and_check_their_agreement():
    runtime = metal_runtime()
    first, second = vector_add_inputs(73, 2, 4097)
    vector = CheckedLaunch(vector_launch(4097, 256), runtime.limits["vector_add"])
    image = uniform_rgb_image(73, 3, 31, 47)
    gray = CheckedLaunch(image_launch(31, 47, (16, 16)), runtime.limits["rgb_to_gray"])

    for workload in (
        runtime.vector_add_workload(first, second, vector),
        runtime.rgb_to_gray_workload(image, gray),
    ):
        workload.metal()
        workload.torch()
        workload.synchronize()
        assert workload.check()


@requires_metal
def test_launches_must_be_checked_against_the_kernels_own_pipeline():
    runtime = metal_runtime()
    first, second = vector_add_inputs(73, 0, 64)
    other = PipelineLimits(512, 32, 0)

    for launch in (
        CheckedLaunch(vector_launch(64, 64), other),
        CheckedLaunch(image_launch(8, 8, (8, 8)), runtime.limits["vector_add"]),
    ):
        with pytest.raises(KernelMetalError, match="checked against its own pipeline"):
            runtime.vector_add(first, second, launch)
    launch = CheckedLaunch(vector_launch(64, 64), runtime.limits["vector_add"])
    with pytest.raises(KernelMetalError, match="launch length"):
        runtime.vector_add(first[:32], second[:32], launch)
    with pytest.raises(KernelMetalError, match="device must be one of"):
        runtime.torch_vector_add(first, second, "cuda")
