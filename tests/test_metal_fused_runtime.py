import functools
import re
from types import SimpleNamespace

import numpy
import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.metal_fused_plan import TORCH_SOFTMAX
from ai_journey.metal_fused_reference import (
    CORRECTNESS_GAP,
    EXP_PROBE_COUNT,
    NORMAL,
    UNIFORM_80,
    RowLayout,
    dense_layout,
    error_bounds,
    exp_probe_arguments,
    exp_probe_errors,
    ragged_layout,
    softmax_inputs,
    softmax_reference,
)
from ai_journey.metal_fused_runtime import (
    FEWER_THREADS,
    OTHER_SIMD_WIDTH,
    OTHER_THREADGROUP_BYTES,
    FusedAdmission,
    MetalFusedRuntime,
    MetalFusedUnavailableError,
    admit,
    checked_launch,
)
from ai_journey.metal_fused_source import (
    EXP_PROBE,
    F1,
    F2,
    F3,
    KERNELS,
    STEP_VARIANTS,
    U0,
    U0_EXP_SUM,
    U0_ROW_MAX,
    MetalFusedError,
)
from test_kernel_metal_runtime import darwin, fake_torch, importer, requires_metal

RAGGED = RowLayout((1, 31, 33, 1000, 4096, 2049, 7), CORRECTNESS_GAP)


@functools.cache
def fused_runtime():
    """Compile the library once per test process; call only under requires_metal."""

    return MetalFusedRuntime()


def fake_pipelines(threads=None, width=None, memory=None):
    """A torch stand-in whose library reports chosen pipeline limits."""

    threads, width, memory = threads or {}, width or {}, memory or {}

    def compile_shader(source):
        pipelines = {
            name: SimpleNamespace(
                max_threads_per_threadgroup=threads.get(name, 1024),
                thread_execution_width=width.get(name, 32),
                static_thread_group_memory_length=memory.get(
                    name, KERNELS[name].threadgroup_bytes
                ),
            )
            for name in re.findall(r"kernel void (\w+)\(", source)
        }
        return SimpleNamespace(**pipelines)

    torch = fake_torch()
    torch.mps = SimpleNamespace(compile_shader=compile_shader)
    return torch


def fake_runtime(environ=None, **limits):
    return MetalFusedRuntime(
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
    with pytest.raises(MetalFusedUnavailableError, match=reason):
        MetalFusedRuntime(system=darwin, import_module=importer(module))
    with pytest.raises(MetalFusedUnavailableError, match="unsupported_operating"):
        MetalFusedRuntime(
            system=lambda: "Linux",
            import_module=importer(AssertionError("torch must not be imported")),
        )


def test_a_rejected_library_is_named_as_an_unavailable_host():
    torch = fake_pipelines()

    def refuse(source):
        raise RuntimeError("program_source:12: error: use of undeclared identifier")

    torch.mps = SimpleNamespace(compile_shader=refuse)
    with pytest.raises(MetalFusedUnavailableError, match="fused softmax library"):
        MetalFusedRuntime(system=darwin, import_module=importer(torch), environ={})


def test_the_runtime_reads_every_pipeline_and_copies_the_switches():
    runtime = fake_runtime(
        environ={"PYTORCH_MPS_FAST_MATH": "1", "PYTORCH_MPS_PREFER_METAL": "a b"}
    )

    assert set(runtime.limits) == set(KERNELS)
    assert runtime.limits[F1] == PipelineLimits(1024, 32, 8192)
    assert runtime.mps_switches == {
        "PYTORCH_MPS_FAST_MATH": "1",
        "PYTORCH_MPS_PREFER_METAL": "unknown",
    }
    assert fake_runtime().mps_switches == {
        "PYTORCH_MPS_FAST_MATH": None,
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    assert all(entry.admitted for entry in admit(runtime.limits, 1024).values())


@pytest.mark.parametrize(
    ("limits", "variant", "reason"),
    [
        ({"threads": {U0_EXP_SUM: 512}}, U0, FEWER_THREADS),
        ({"threads": {F3: 896}}, F3, FEWER_THREADS),
        ({"width": {F2: 64}}, F2, OTHER_SIMD_WIDTH),
        ({"memory": {U0_ROW_MAX: 4112}}, U0, OTHER_THREADGROUP_BYTES),
        ({"memory": {F1: 4096}}, F1, OTHER_THREADGROUP_BYTES),
    ],
)
def test_a_pipeline_refuses_its_step_with_the_first_reason(limits, variant, reason):
    runtime = fake_runtime(**limits)
    verdicts = admit(runtime.limits, 1024)

    assert verdicts[variant].reason == reason
    assert [name for name, entry in verdicts.items() if not entry.admitted] == [variant]
    mapping = verdicts[variant].to_mapping()
    assert mapping["admitted"] is False and mapping["required_threads"] == 1024
    assert [kernel["name"] for kernel in mapping["kernels"]] == list(
        verdicts[variant].limits
    )
    with pytest.raises(MetalFusedError, match="refused by its pipeline"):
        runtime.run(variant, numpy.zeros(4, numpy.float32), dense_layout(1, 4), 1024)


def test_a_smaller_run_admits_a_pipeline_that_allows_fewer_threads():
    runtime = fake_runtime(threads={F3: 256})

    assert admit(runtime.limits, 256)[F3].admitted
    assert not admit(runtime.limits, 512)[F3].admitted
    with pytest.raises(MetalFusedError, match="limits of each kernel"):
        FusedAdmission(F1, 32, {U0_ROW_MAX: runtime.limits[U0_ROW_MAX]})


def test_each_row_launch_is_one_group_per_row_on_its_own_pipeline():
    runtime = fake_runtime(threads={F2: 128})
    layout = dense_layout(5, 1000)

    launch = checked_launch(runtime.limits, F1, layout)
    assert launch.geometry.launch_arguments() == {"threads": 5 * 64, "group_size": 64}
    assert launch.geometry.groups == (5,)
    assert launch.simd_groups_per_threadgroup == 2
    with pytest.raises(MetalFusedError, match="exceeds this pipeline"):
        checked_launch(runtime.limits, F2, dense_layout(2, 4096))
    with pytest.raises(MetalFusedError, match="not a row kernel"):
        checked_launch(runtime.limits, EXP_PROBE, layout)


def test_the_runtime_refuses_launches_outside_the_run():
    runtime = fake_runtime()
    layout = dense_layout(2, 4096)
    values = numpy.zeros(layout.buffer_elements, numpy.float32)

    with pytest.raises(MetalFusedError, match="more threads than the run admitted"):
        runtime.run(F1, values, layout, 128)
    with pytest.raises(MetalFusedError, match="registered step"):
        runtime.run(TORCH_SOFTMAX, values, layout, 1024)
    with pytest.raises(MetalFusedError, match="no gap"):
        runtime.workload(values, RowLayout((3,), 1), (U0,))
    with pytest.raises(MetalFusedError, match="plan order"):
        runtime.workload(values, layout, (F1, U0))
    with pytest.raises(MetalFusedError, match="plan order"):
        runtime.workload(values, layout, ())
    with pytest.raises(MetalFusedError, match="multiple of 256"):
        runtime.exp_probe(numpy.zeros(100, numpy.float32))


@requires_metal
def test_this_gpu_admits_every_step_for_the_largest_group():
    runtime = fused_runtime()

    for name, limits in runtime.limits.items():
        assert limits.max_threads_per_threadgroup == 1024, name
        assert limits.thread_execution_width == 32, name
        assert limits.static_threadgroup_memory_bytes == (
            KERNELS[name].threadgroup_bytes
        ), name
    assert all(entry.admitted for entry in admit(runtime.limits, 1024).values())


@requires_metal
@pytest.mark.parametrize("variant", STEP_VARIANTS)
def test_each_step_meets_its_bound_and_keeps_every_gap_on_this_gpu(variant):
    runtime = fused_runtime()
    values = softmax_inputs(79, 0, 9, RAGGED, UNIFORM_80)
    reference = softmax_reference(values, RAGGED)

    output = runtime.run(variant, values, RAGGED, RAGGED.group_size)

    gaps = numpy.ones(RAGGED.buffer_elements, dtype=bool)
    gaps[RAGGED.valid()] = False
    assert numpy.all(output[gaps].view(numpy.uint32) == 0x7FC00000)
    packed = output[RAGGED.valid()].astype(numpy.float64)
    assert numpy.all(numpy.isfinite(packed))
    bounds = error_bounds(variant, reference).elements
    assert numpy.all(numpy.abs(packed - reference.output) <= bounds)


@requires_metal
def test_the_exp_probe_meets_its_rule_on_this_gpu():
    results = fused_runtime().exp_probe(exp_probe_arguments())

    probe = exp_probe_errors(results)
    assert results.size == EXP_PROBE_COUNT
    assert probe.holds and probe.max_ulps <= 4


@requires_metal
@pytest.mark.parametrize(
    "layout", [dense_layout(64, 300), ragged_layout(79, 0, 40, 2000)], ids=str
)
def test_a_workload_runs_every_side_on_one_set_of_device_rows(layout):
    runtime = fused_runtime()
    values = softmax_inputs(79, 1, 0, layout, NORMAL)
    reference = softmax_reference(values, layout)
    names = (U0, F1, F2, F3)

    workload = runtime.workload(values, layout, names)
    assert tuple(workload.sides) == (*names, TORCH_SOFTMAX)
    for call in workload.calls:
        call()
    outputs = workload.outputs()

    for name in names:
        error = numpy.abs(outputs[name].astype(numpy.float64) - reference.output)
        assert numpy.all(error <= error_bounds(name, reference).elements), name
    torch_error = numpy.abs(outputs[TORCH_SOFTMAX] - reference.output)
    assert outputs[TORCH_SOFTMAX].shape == (layout.elements,)
    assert float(torch_error.max()) < 2.0**-16
