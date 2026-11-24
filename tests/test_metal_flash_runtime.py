import functools
import re
from types import SimpleNamespace

import numpy
import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.metal_flash_plan import SIDES
from ai_journey.metal_flash_reference import (
    CORRECTNESS_GAP,
    FlashLayout,
    dense_layout,
    flash_inputs,
    flash_reference,
    gather_rows,
)
from ai_journey.metal_flash_runtime import (
    COUNTERS,
    FEWER_THREADS,
    OTHER_SIMD_WIDTH,
    OTHER_THREADGROUP_BYTES,
    READINGS,
    FlashAdmission,
    MetalFlashRuntime,
    MetalFlashUnavailableError,
    admit,
    launch_mappings,
    launches,
)
from ai_journey.metal_flash_source import (
    CAUSAL,
    F1,
    FULL,
    KERNELS,
    MODES,
    N0,
    VARIANTS,
    MetalFlashError,
)
from ai_journey.metal_fused_reference import SENTINEL_BITS, exp_probe_arguments
from ai_journey.metal_fused_reference import exp_probe_errors
from test_kernel_metal_runtime import darwin, fake_torch, importer, requires_metal

RAGGED = FlashLayout((1, 31, 33, 100), CORRECTNESS_GAP)


@functools.cache
def flash_runtime():
    """Compile the library once per test process; call only under requires_metal."""

    return MetalFlashRuntime()


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


def test_a_host_that_is_not_macos_is_refused_before_torch_is_imported():
    with pytest.raises(MetalFlashUnavailableError, match="only on macOS"):
        MetalFlashRuntime(
            system=lambda: "Linux",
            import_module=importer(AssertionError("torch must not be imported")),
        )


def test_a_compiler_error_makes_the_host_unavailable():
    torch = fake_torch()

    def reject(source):
        raise RuntimeError("bad text")

    torch.mps = SimpleNamespace(compile_shader=reject)
    with pytest.raises(MetalFlashUnavailableError, match="rejected"):
        MetalFlashRuntime(system=darwin, import_module=importer(torch))


def test_the_runtime_reads_every_pipeline_and_the_switches():
    runtime = MetalFlashRuntime(
        system=darwin,
        import_module=importer(fake_pipelines()),
        environ={"PYTORCH_MPS_FAST_MATH": "1"},
    )

    assert set(runtime.limits) == set(KERNELS)
    assert runtime.mps_switches == {
        "PYTORCH_MPS_FAST_MATH": "1",
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    assert all(verdict.admitted for verdict in admit(runtime.limits).values())


@pytest.mark.parametrize(
    ("limits", "reason"),
    [
        (PipelineLimits(64, 32, 16384), FEWER_THREADS),
        (PipelineLimits(1024, 64, 16384), OTHER_SIMD_WIDTH),
        (PipelineLimits(1024, 32, 8192), OTHER_THREADGROUP_BYTES),
        (PipelineLimits(1024, 32, 16384), None),
    ],
)
def test_a_pipeline_admits_only_the_group_width_and_memory_of_its_text(limits, reason):
    admission = FlashAdmission("flash_f1", limits)

    assert admission.reason == reason
    assert admission.to_mapping()["required_threads"] == 128
    assert admission.to_mapping()["declared_threadgroup_bytes"] == 16384


def test_a_refused_pipeline_stops_its_side():
    runtime = MetalFlashRuntime(
        system=darwin,
        import_module=importer(fake_pipelines(threads={"flash_n0_pv": 32})),
    )
    inputs = flash_inputs(81, 0, 0, RAGGED, "normal")

    with pytest.raises(MetalFlashError, match="flash_n0_pv was refused"):
        runtime.run(inputs, N0, FULL)


def test_each_side_launches_one_group_per_query_row_or_query_block():
    n0 = launches(N0, CAUSAL, RAGGED)
    assert [name for name, _ in n0] == [
        "flash_n0_scores_causal",
        "flash_n0_softmax_causal",
        "flash_n0_pv_causal",
    ]
    assert [geometry.groups[0] for _, geometry in n0] == [165, 165, 165]
    assert [geometry.group_size[0] for _, geometry in n0] == [256, 256, 64]
    ((name, geometry),) = launches(F1, FULL, RAGGED)
    assert (name, geometry.groups[0], geometry.group_size[0]) == ("flash_f1", 8, 128)
    assert launch_mappings(F1, FULL, RAGGED)[0]["padding_threads"] == 0


@requires_metal
def test_this_gpu_admits_every_kernel_of_the_library():
    runtime = flash_runtime()

    for name, kernel in KERNELS.items():
        limits = runtime.limits[name]
        assert limits.thread_execution_width == 32
        assert limits.static_threadgroup_memory_bytes == kernel.threadgroup_bytes
        assert admit(runtime.limits)[name].admitted


@requires_metal
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_this_gpu_keeps_the_gap_rows_and_the_bound(variant, mode):
    runtime = flash_runtime()
    inputs = flash_inputs(81, 0, 0, RAGGED, "wide_scores")
    reference = flash_reference(inputs, mode)

    output = runtime.run(inputs, variant, mode)
    gaps = numpy.ones(RAGGED.buffer_rows, dtype=bool)
    gaps[RAGGED.valid_rows()] = False
    assert numpy.all(output.reshape(-1, 64)[gaps].view(numpy.uint32) == SENTINEL_BITS)
    error = numpy.abs(gather_rows(output, reference.rows) - reference.output)
    assert numpy.max(error / reference.bound) <= 1.0


@requires_metal
def test_this_gpu_meets_the_exp_probe_rule():
    runtime = flash_runtime()

    assert exp_probe_errors(runtime.exp_probe(exp_probe_arguments())).holds


@requires_metal
@pytest.mark.parametrize("mode", MODES)
def test_this_gpu_binds_three_sides_and_reads_their_memory(mode):
    runtime = flash_runtime()
    inputs = flash_inputs(81, 1, 0, dense_layout(2, 96), "normal")
    reference = flash_reference(inputs, mode, 8)

    workload = runtime.workload(inputs, mode)
    assert tuple(workload.sides) == SIDES
    for call in workload.calls:
        call()
    outputs = workload.outputs()
    for side in SIDES:
        error = numpy.abs(gather_rows(outputs[side], reference.rows) - reference.output)
        assert numpy.max(error / reference.vendor_bound) <= 1.0
    readings = runtime.memory(inputs, mode)
    assert tuple(readings) == SIDES
    for entry in readings.values():
        assert set(entry) == {*READINGS, "polls"}
        assert entry["polls"] >= 1
        assert all(set(entry[name]) == set(COUNTERS) for name in READINGS)


@requires_metal
def test_this_gpu_refuses_a_gapped_timing_point():
    inputs = flash_inputs(81, 1, 0, RAGGED, "normal")

    with pytest.raises(MetalFlashError, match="dense"):
        flash_runtime().workload(inputs, FULL)
