import functools
import re
from types import SimpleNamespace

import numpy
import pytest

from ai_journey.gpu_ceiling_source import STREAM_COPY
from ai_journey.gpu_scan_plan import COPY, VENDOR
from ai_journey.gpu_scan_reference import (
    BENCHMARK_STREAM,
    NORMAL,
    REPLAYS,
    error_bound,
    float64_reference,
    max_relative_error,
    scan_inputs,
)
from ai_journey.gpu_scan_runtime import (
    COPY_GROUP,
    FEWER_THREADS,
    OTHER_SIMD_WIDTH,
    OTHER_THREADGROUP_BYTES,
    GpuScanUnavailableError,
    ScanRuntime,
    copy_float4,
    copy_launch,
    padded_copy_input,
    scan_plan,
)
from ai_journey.gpu_scan_source import (
    ADD_OFFSETS,
    KERNELS,
    S0_HILLIS_STEELE,
    S1_BLELLOCH,
    S2_SIMD,
    GpuScanError,
    guard_elements,
    library_source,
)
from ai_journey.kernel_metal_reference import FLOAT32_SENTINEL_BITS
from test_kernel_metal_runtime import darwin, fake_torch, importer, requires_metal


@functools.cache
def scan_runtime():
    """Compile both libraries once per test process; call only under requires_metal."""

    return ScanRuntime()


def fake_pipelines(threads=None, memory=None, widths=None, refuse=None):
    """A torch stand-in whose libraries report chosen pipeline limits."""

    threads, memory, widths = threads or {}, memory or {}, widths or {}

    def compile_shader(source):
        if refuse is not None and refuse in source:
            raise RuntimeError("program_source:9: error: unknown type name")
        pipelines = {}
        for name in re.findall(r"kernel void (\w+)\(", source):
            label = COPY if name == STREAM_COPY else name
            declared = KERNELS[name].threadgroup_bytes if name in KERNELS else 0
            pipelines[name] = SimpleNamespace(
                max_threads_per_threadgroup=threads.get(label, 1024),
                thread_execution_width=widths.get(label, 32),
                static_thread_group_memory_length=memory.get(label, declared),
            )
        return SimpleNamespace(**pipelines)

    torch = fake_torch()
    torch.mps = SimpleNamespace(compile_shader=compile_shader)
    return torch


def fake_runtime(environ=None, **limits):
    return ScanRuntime(
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
    with pytest.raises(GpuScanUnavailableError, match=reason):
        ScanRuntime(system=darwin, import_module=importer(module))
    with pytest.raises(GpuScanUnavailableError, match="unsupported_operating"):
        ScanRuntime(
            system=lambda: "Linux",
            import_module=importer(AssertionError("torch must not be imported")),
        )


@pytest.mark.parametrize(
    ("marker", "label"),
    [
        ("kernel void scan_s2_simd(", "the scan library"),
        ("kernel void stream_copy(", "the Day 74 ceiling source"),
    ],
)
def test_a_rejected_library_is_named_as_an_unavailable_host(marker, label):
    with pytest.raises(GpuScanUnavailableError, match=label):
        fake_runtime(refuse=marker)


def test_each_pipeline_verdict_is_data_and_a_refused_kernel_never_runs():
    runtime = fake_runtime(
        threads={S1_BLELLOCH: 256},
        memory={S0_HILLIS_STEELE: 4096},
        widths={S2_SIMD: 64},
        environ={"PYTORCH_MPS_FAST_MATH": "1", "PYTORCH_MPS_PREFER_METAL": "x y"},
    )

    reasons = {name: entry.reason for name, entry in runtime.admissions.items()}
    assert reasons == {
        S0_HILLIS_STEELE: OTHER_THREADGROUP_BYTES,
        S1_BLELLOCH: FEWER_THREADS,
        S2_SIMD: OTHER_SIMD_WIDTH,
    }
    assert runtime.admissions[S2_SIMD].to_mapping() == {
        "name": S2_SIMD,
        "admitted": False,
        "reason": OTHER_SIMD_WIDTH,
        "required_threads": 1024,
        "allowed_threads": 1024,
        "declared_threadgroup_bytes": 128,
        "static_threadgroup_bytes": 128,
        "required_simd_width": 32,
        "simd_width": 64,
    }
    assert runtime.mps_switches == {
        "PYTORCH_MPS_FAST_MATH": "1",
        "PYTORCH_MPS_PREFER_METAL": "unknown",
    }
    assert runtime.library_source == library_source()
    assert set(runtime.limits) == {*KERNELS, ADD_OFFSETS, COPY}
    for name in KERNELS:
        with pytest.raises(GpuScanError, match="refused by its pipeline"):
            runtime.scan(name, numpy.ones(8, numpy.float32))


@pytest.mark.parametrize("helper", [ADD_OFFSETS, COPY])
def test_a_helper_pipeline_without_groups_of_256_makes_the_host_unavailable(helper):
    with pytest.raises(GpuScanUnavailableError, match="fewer than 256 threads"):
        fake_runtime(threads={helper: 128})


def test_each_level_launches_whole_blocks_and_each_add_one_thread_per_element():
    limits = fake_runtime().limits

    plan = scan_plan(S1_BLELLOCH, limits, 1048577)
    rows = [
        (
            level.length,
            level.scan.geometry.extent,
            level.scan.geometry.group_size,
            None if level.add is None else level.add.geometry.threads,
        )
        for level in plan
    ]

    assert rows == [
        (1048577, (1025 * 512,), (512,), (1048832,)),
        (1025, (2 * 512,), (512,), (1280,)),
        (2, (512,), (512,), None),
    ]
    assert plan[0].to_mapping()["blocks"] == 1025
    assert plan[0].to_mapping()["scan"]["padding_threads"] == 0
    assert scan_plan(S0_HILLIS_STEELE, limits, 1)[0].scan.geometry.extent == (1024,)
    with pytest.raises(GpuScanError, match="not a registered"):
        scan_plan("scan_s9_other", limits, 8)


def test_the_copy_walks_ceil_n_over_4_float4_values_four_for_each_thread():
    limits = fake_runtime().limits
    values = numpy.arange(1, 6, dtype=numpy.float32)

    launch = copy_launch(limits, 5000003)

    assert copy_float4(5000003) == 1250001
    assert launch.geometry.extent == (312501,)
    assert launch.geometry.group_size == (COPY_GROUP,)
    assert padded_copy_input(values).tolist() == [1, 2, 3, 4, 5, 0, 0, 0]
    assert padded_copy_input(values[:4]).tolist() == [1, 2, 3, 4]


@requires_metal
def test_this_gpu_admits_every_kernel_with_its_declared_memory():
    runtime = scan_runtime()

    for name, kernel in KERNELS.items():
        assert runtime.admissions[name].admitted, runtime.admissions[name].reason
        limits = runtime.limits[name]
        assert limits.static_threadgroup_memory_bytes == kernel.threadgroup_bytes
        assert limits.thread_execution_width == 32
    assert runtime.torch_version is not None


@requires_metal
@pytest.mark.parametrize("length", [1, 1000, 1025, 70001])
def test_this_gpu_scans_within_the_bound_and_keeps_every_tail(length):
    runtime = scan_runtime()
    values = scan_inputs(78, BENCHMARK_STREAM, 7, length, NORMAL)
    reference, scale = float64_reference(values)

    for name in KERNELS:
        buffer = runtime.scan(name, values)
        assert buffer.shape == (length + guard_elements(length),)
        assert (buffer[length:].view(numpy.uint32) == FLOAT32_SENTINEL_BITS).all()
        error = max_relative_error(buffer[:length], reference, scale)
        assert error <= error_bound(name, length), name
        if name in REPLAYS:
            replay = REPLAYS[name](values).view(numpy.uint32)
            assert numpy.array_equal(buffer[:length].view(numpy.uint32), replay)
    vendor = runtime.cumsum(values)
    assert vendor.shape == (length,)
    assert max_relative_error(vendor, reference, scale) < 1e-5


@requires_metal
def test_a_workload_binds_every_side_and_returns_each_latest_output():
    runtime = scan_runtime()
    values = scan_inputs(78, BENCHMARK_STREAM, 8, 5003, NORMAL)

    workload = runtime.workload(values, tuple(KERNELS))
    for call in workload.calls:
        call()
    outputs = workload.check()

    assert list(workload.sides) == [*KERNELS, VENDOR, COPY]
    assert numpy.array_equal(outputs[COPY], padded_copy_input(values))
    for name in REPLAYS:
        assert numpy.array_equal(outputs[name], REPLAYS[name](values))
    reference, scale = float64_reference(values)
    assert max_relative_error(outputs[VENDOR], reference, scale) < 1e-5
    with pytest.raises(GpuScanError, match="registry order"):
        runtime.workload(values, (S2_SIMD, S0_HILLIS_STEELE))
