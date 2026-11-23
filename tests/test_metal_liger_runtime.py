import functools
import re
from types import SimpleNamespace

import numpy
import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.metal_fused_reference import SENTINEL_BITS, exp_probe_arguments
from ai_journey.metal_fused_reference import exp_probe_errors
from ai_journey.metal_liger_plan import TORCH_SIDES
from ai_journey.metal_liger_reference import (
    CORRECTNESS_GAP,
    LigerLayout,
    dense_layout,
    liger_inputs,
    liger_reference,
    log_probe_arguments,
    math_probe_errors,
    rsqrt_probe_arguments,
)
from ai_journey.metal_liger_runtime import (
    FEWER_THREADS,
    OTHER_SIMD_WIDTH,
    OTHER_THREADGROUP_BYTES,
    LigerAdmission,
    MetalLigerRuntime,
    MetalLigerUnavailableError,
    admit,
    checked_launch,
)
from ai_journey.metal_liger_source import (
    CROSS_ENTROPY,
    CROSS_ENTROPY_KERNEL,
    FAMILIES,
    FAMILY_KERNELS,
    KERNELS,
    PROBE_EXP,
    RMSNORM,
    RMSNORM_KERNEL,
    SWIGLU,
    MetalLigerError,
)
from test_kernel_metal_runtime import darwin, fake_torch, importer, requires_metal

RAGGED = {
    RMSNORM: LigerLayout((1, 31, 33, 1000, 4096, 16384, 7), CORRECTNESS_GAP),
    SWIGLU: LigerLayout((1, 31, 33, 1000, 4096, 7), CORRECTNESS_GAP),
    CROSS_ENTROPY: LigerLayout((1, 31, 33, 1000, 50257, 7), CORRECTNESS_GAP),
}


@functools.cache
def liger_runtime():
    """Compile the library once per test process; call only under requires_metal."""

    return MetalLigerRuntime()


def fake_pipelines(threads=None, width=None, memory=None):
    """A torch stand-in whose library reports chosen pipeline limits."""

    threads, width, memory = threads or {}, width or {}, memory or {}

    def compile_shader(source):
        return SimpleNamespace(
            **{
                name: SimpleNamespace(
                    max_threads_per_threadgroup=threads.get(name, 1024),
                    thread_execution_width=width.get(name, 32),
                    static_thread_group_memory_length=memory.get(
                        name, KERNELS[name].threadgroup_bytes
                    ),
                )
                for name in re.findall(r"kernel void (\w+)\(", source)
            }
        )

    torch = fake_torch()
    torch.mps = SimpleNamespace(compile_shader=compile_shader)
    return torch


def test_a_host_without_macos_stops_before_torch_is_imported():
    with pytest.raises(MetalLigerUnavailableError, match="unsupported_operating"):
        MetalLigerRuntime(
            system=lambda: "Linux",
            import_module=importer(AssertionError("torch must not be imported")),
        )


def test_a_rejected_library_is_an_unavailable_host():
    torch = fake_torch()

    def reject(source):
        raise RuntimeError("bad text")

    torch.mps = SimpleNamespace(compile_shader=reject)
    with pytest.raises(MetalLigerUnavailableError, match="rejected the Day 80"):
        MetalLigerRuntime(system=darwin, import_module=importer(torch))


def test_the_runtime_reads_every_pipeline_and_the_mps_switches():
    runtime = MetalLigerRuntime(
        system=darwin,
        import_module=importer(fake_pipelines(threads={RMSNORM_KERNEL: 512})),
        environ={"PYTORCH_MPS_FAST_MATH": "1"},
    )

    assert set(runtime.limits) == set(KERNELS)
    assert runtime.limits[RMSNORM_KERNEL].max_threads_per_threadgroup == 512
    assert runtime.mps_switches == {
        "PYTORCH_MPS_FAST_MATH": "1",
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    assert runtime.torch_version == "2.15.0"


def test_every_refusal_names_the_first_limit_that_fails():
    good = {
        name: PipelineLimits(1024, 32, kernel.threadgroup_bytes)
        for name, kernel in KERNELS.items()
    }
    verdicts = admit(good, 1024)
    assert all(verdict.admitted for verdict in verdicts.values())
    assert verdicts[PROBE_EXP].required_threads == 256
    assert verdicts[RMSNORM_KERNEL].to_mapping()["declared_threadgroup_bytes"] == 128

    assert LigerAdmission(
        RMSNORM_KERNEL, 1024, PipelineLimits(512, 32, 128)
    ).reason == (FEWER_THREADS)
    assert LigerAdmission(RMSNORM_KERNEL, 64, PipelineLimits(512, 16, 128)).reason == (
        OTHER_SIMD_WIDTH
    )
    assert LigerAdmission(
        CROSS_ENTROPY_KERNEL, 64, PipelineLimits(512, 32, 128)
    ).reason == (OTHER_THREADGROUP_BYTES)
    with pytest.raises(MetalLigerError, match="registered kernel"):
        LigerAdmission("liger_layernorm", 32, PipelineLimits(512, 32, 0))


def test_a_launch_is_one_group_of_g_threads_for_each_row():
    limits = {
        name: PipelineLimits(1024, 32, kernel.threadgroup_bytes)
        for name, kernel in KERNELS.items()
    }
    launch = checked_launch(limits, CROSS_ENTROPY, dense_layout(1024, 50257))

    assert launch.geometry.launch_arguments() == {
        "threads": 1024 * 1024,
        "group_size": 1024,
    }
    small = {**limits, RMSNORM_KERNEL: PipelineLimits(128, 32, 128)}
    with pytest.raises(MetalLigerError, match="rmsnorm"):
        checked_launch(small, RMSNORM, dense_layout(2, 4096))


@requires_metal
def test_this_gpu_admits_every_kernel_with_its_declared_bytes():
    runtime = liger_runtime()

    assert all(verdict.admitted for verdict in admit(runtime.limits, 1024).values())


@requires_metal
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("distribution", ["uniform_80", "offset_plus_1000"])
def test_each_kernel_keeps_its_gaps_and_bound_on_the_apple_gpu(family, distribution):
    runtime = liger_runtime()
    layout = RAGGED[family]
    inputs = liger_inputs(80, family, 0, 5, layout, distribution)
    reference = liger_reference(inputs)

    first = runtime.run(inputs, 1024)
    second = runtime.run(inputs, 1024)
    valid = layout.output_valid(family)
    gaps = numpy.ones(first.size, dtype=bool)
    gaps[valid] = False
    assert numpy.all(first[gaps].view(numpy.uint32) == SENTINEL_BITS)
    error = numpy.abs(first[valid].astype(numpy.float64) - reference.output)
    assert numpy.max(error / reference.bound) <= 1.0
    assert numpy.array_equal(first.view(numpy.uint32), second.view(numpy.uint32))


@requires_metal
def test_the_precise_math_meets_the_assumptions_of_the_bounds():
    runtime = liger_runtime()

    assert exp_probe_errors(runtime.probe("exp", exp_probe_arguments())).holds
    assert math_probe_errors(
        "rsqrt", runtime.probe("rsqrt", rsqrt_probe_arguments())
    ).holds
    assert math_probe_errors("log", runtime.probe("log", log_probe_arguments())).holds
    with pytest.raises(MetalLigerError, match="no probe kernel"):
        runtime.probe("tanh", rsqrt_probe_arguments())


@requires_metal
@pytest.mark.parametrize("family", FAMILIES)
def test_a_workload_binds_the_kernel_and_its_torch_sequence(family):
    runtime = liger_runtime()
    layout = dense_layout(64, 1000)
    inputs = liger_inputs(80, family, 1, 0, layout, "normal")
    workload = runtime.workload(inputs)

    assert tuple(workload.sides) == (FAMILY_KERNELS[family], TORCH_SIDES[family])
    for call in workload.calls:
        call()
    outputs = workload.outputs()
    reference = liger_reference(inputs)
    kernel = outputs[FAMILY_KERNELS[family]].astype(numpy.float64)
    vendor = outputs[TORCH_SIDES[family]].astype(numpy.float64)
    assert numpy.max(numpy.abs(kernel - reference.output) / reference.bound) <= 1
    assert numpy.max(numpy.abs(vendor - reference.output) / reference.vendor_bound) <= 1
    with pytest.raises(MetalLigerError, match="dense rows with no gap"):
        runtime.workload(liger_inputs(80, family, 1, 0, RAGGED[family], "normal"))
