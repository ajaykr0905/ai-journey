import functools
import re
from types import SimpleNamespace

import numpy
import pytest

from ai_journey import metal_tc_runtime
from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.metal_tc_plan import SIDES
from ai_journey.metal_tc_reference import (
    SENTINEL_BITS,
    TAIL,
    operands,
    tc_reference,
)
from ai_journey.metal_tc_runtime import (
    CALL_FAILED,
    COMPILE_FAILED,
    FEWER_THREADS,
    OTHER_SIMD_WIDTH,
    OTHER_THREADGROUP_BYTES,
    MetalTcRuntime,
    MetalTcUnavailableError,
    Support,
    TcAdmission,
    admit,
    launch,
    launch_mapping,
    needed_precisions,
    support_operands,
)
from ai_journey.metal_tc_source import (
    BFLOAT_LIBRARY,
    KERNEL_NAME,
    KERNELS,
    M1,
    M3,
    MAIN_LIBRARY,
    PRECISION,
    VARIANTS,
    MetalTcError,
    library_source,
)
from test_kernel_metal_runtime import darwin, fake_torch, importer, requires_metal

RAGGED = MatmulShape(45, 70, 37)


@functools.cache
def tc_runtime():
    """Compile both libraries once per test process; call only under requires_metal."""

    return MetalTcRuntime()


class _Probe:
    """The output of the version probe in the torch stand-in."""

    value = 0

    def cpu(self):
        return self

    def item(self):
        return self.value


class _Pipeline:
    """One kernel of the torch stand-in, with the limits its pipeline reports."""

    def __init__(self, name):
        self.max_threads_per_threadgroup = 1024
        self.thread_execution_width = 32
        self.static_thread_group_memory_length = KERNELS[name].threadgroup_bytes

    def __call__(self, out, **launch_arguments):
        out.value = 400


class _Library:
    """A compiled library of the stand-in; with ``broken`` no kernel builds."""

    def __init__(self, source, broken):
        self.names = re.findall(r"kernel void (\w+)\(", source)
        self.broken = broken

    def __getattr__(self, name):
        if name not in self.names:
            raise AttributeError(name)
        if self.broken is not None:
            raise self.broken
        return _Pipeline(name)


def refusing_torch(library, error, stage="compile"):
    """A torch stand-in that refuses one library at one stage, as PyTorch does.

    PyTorch raises ``SyntaxError`` when the Metal compiler rejects a text, and
    ``RuntimeError`` when a kernel of a compiled library builds no pipeline,
    because it builds the pipeline when the kernel is read. The stand-in
    makes no tensor, so its bfloat16 call fails too.
    """

    def compile_shader(source):
        refused = source == library_source(library)
        if refused and stage == "compile":
            raise error
        return _Library(source, error if refused else None)

    def no_tensor(values):
        raise RuntimeError("the stand-in makes no tensor")

    torch = fake_torch()
    torch.mps = SimpleNamespace(compile_shader=compile_shader, synchronize=print)
    for dtype in ("int32", "float32", "float16", "bfloat16"):
        setattr(torch, dtype, dtype)
    torch.zeros = lambda *shape, **options: _Probe()
    torch.from_numpy = no_tensor
    return torch


@pytest.mark.parametrize(
    ("error", "stage"),
    [
        (SyntaxError("program_source: error: unknown type name 'bfloat'"), "compile"),
        (RuntimeError("Failed to create pipeline state object"), "pipeline"),
    ],
)
def test_a_host_that_refuses_the_bfloat_library_runs_the_other_kernels(error, stage):
    torch = refusing_torch(BFLOAT_LIBRARY, error, stage)
    runtime = MetalTcRuntime(system=darwin, import_module=importer(torch))

    assert runtime.support[M3] == Support(False, COMPILE_FAILED)
    assert set(runtime.limits) == set(KERNELS) - {KERNEL_NAME[M3]}
    assert all(verdict.admitted for verdict in admit(runtime.limits).values())
    assert runtime.metal_version == 400
    assert runtime.supported(M1) and not runtime.supported(M3)


def test_a_compiler_that_refuses_the_main_library_makes_the_host_unavailable():
    torch = refusing_torch(MAIN_LIBRARY, SyntaxError("program_source: error"))

    with pytest.raises(MetalTcUnavailableError, match="rejected the Day 82 main"):
        MetalTcRuntime(system=darwin, import_module=importer(torch))


def test_a_host_that_is_not_macos_is_refused_before_torch_is_imported():
    with pytest.raises(MetalTcUnavailableError, match="only on macOS"):
        MetalTcRuntime(
            system=lambda: "Linux",
            import_module=importer(AssertionError("torch must not be imported")),
        )


def test_each_pipeline_refusal_has_its_reason():
    name = KERNEL_NAME[M1]
    declared = KERNELS[name].threadgroup_bytes

    assert TcAdmission(name, PipelineLimits(1024, 32, declared)).admitted
    assert TcAdmission(name, PipelineLimits(64, 32, declared)).reason == FEWER_THREADS
    assert TcAdmission(name, PipelineLimits(1024, 64, declared)).reason == (
        OTHER_SIMD_WIDTH
    )
    assert TcAdmission(name, PipelineLimits(1024, 32, declared + 16)).reason == (
        OTHER_THREADGROUP_BYTES
    )
    limits = {name: PipelineLimits(1024, 32, declared)}
    assert list(admit(limits)) == [name]
    with pytest.raises(MetalTcError):
        TcAdmission("tc_other", PipelineLimits(1024, 32, 0))


def test_a_support_record_needs_one_known_reason():
    assert Support(True, None).to_mapping() == {"supported": True, "reason": None}
    assert Support(False, COMPILE_FAILED).supported is False
    for supported, reason in ((False, None), (True, COMPILE_FAILED), (False, "x")):
        with pytest.raises(MetalTcError):
            Support(supported, reason)


def test_a_support_check_that_raises_records_the_side_as_unsupported():
    runtime = object.__new__(MetalTcRuntime)

    def fail(*args):
        raise RuntimeError("this host makes no bfloat16 tensor")

    runtime._launch = fail
    runtime._tensor = fail

    assert runtime._kernel_support() == Support(False, CALL_FAILED)
    assert runtime._torch_support() == Support(False, CALL_FAILED)


def test_only_the_types_of_supported_sides_reach_the_device():
    without = tuple(side for side in SIDES if side not in ("m3", "torch_bfloat16"))

    assert needed_precisions(SIDES) == ("float32", "float16", "bfloat16")
    assert needed_precisions(without) == ("float32", "float16")


def test_the_support_product_is_exact_in_bfloat():
    a, b, exact = support_operands()

    assert set(numpy.unique(a)) <= {-2.0, -1.0, 0.0, 1.0, 2.0}
    assert numpy.all(numpy.abs(exact) <= 32)
    assert exact.tolist() == (a.astype("float64") @ b.astype("float64")).tolist()


def test_one_group_of_128_threads_runs_for_each_32_by_32_tile():
    geometry = launch(RAGGED)

    assert geometry.groups == (2 * 3,)
    assert geometry.threads_per_group == 128
    assert launch_mapping(MatmulShape(64, 96, 1))["groups"] == [6]
    with pytest.raises(TypeError):
        launch((1, 1, 1))


@requires_metal
def test_this_gpu_compiles_admits_and_reports_its_support():
    runtime = tc_runtime()

    assert set(runtime.limits) == set(KERNELS)
    assert all(verdict.admitted for verdict in admit(runtime.limits).values())
    assert isinstance(runtime.metal_version, int) and runtime.metal_version > 0
    assert set(runtime.support) == {"m3", "torch_bfloat16"}


@requires_metal
def test_every_supported_kernel_meets_its_bound_and_keeps_its_tail():
    runtime = tc_runtime()
    values = operands(82, 0, 0, RAGGED)
    reference = tc_reference(values)
    size = RAGGED.m * RAGGED.n

    for variant in VARIANTS:
        if not runtime.supported(variant):
            continue
        output = runtime.run(values, variant)
        assert output.dtype == numpy.float32 and output.size == size + TAIL
        assert numpy.all(output[size:].view(numpy.uint32) == SENTINEL_BITS)
        error = numpy.abs(output[:size].reshape(45, 70) - reference.output)
        assert numpy.max(error / reference.bound(PRECISION[variant])) <= 1.0


@requires_metal
def test_the_workload_binds_seven_sides_in_plan_order():
    runtime = tc_runtime()
    values = operands(82, 1, 0, MatmulShape(64, 64, 64))
    workload = runtime.workload(values)

    assert tuple(workload.sides) == SIDES
    for call in workload.calls:
        call()
    outputs = workload.outputs()
    assert tuple(outputs) == workload.supported
    reference = tc_reference(values)
    for side, output in outputs.items():
        error = numpy.abs(output.reshape(64, 64) - reference.output)
        assert numpy.max(error / reference.magnitude) < 2.0**-4, side


@requires_metal
def test_this_compiler_refuses_a_broken_bfloat_text_and_m3_is_unsupported(
    monkeypatch,
):
    registry = metal_tc_runtime.library_source

    def broken(library):
        text = registry(library)
        if library == BFLOAT_LIBRARY:
            text += "kernel void tc_broken() { undeclared_value; }\n"
        return text

    monkeypatch.setattr(metal_tc_runtime, "library_source", broken)
    runtime = MetalTcRuntime()

    assert runtime.support[M3] == Support(False, COMPILE_FAILED)
    assert set(runtime.limits) == set(KERNELS) - {KERNEL_NAME[M3]}
    assert runtime.supported(M1)
