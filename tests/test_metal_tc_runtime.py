import functools

import numpy
import pytest

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
    KERNEL_NAME,
    KERNELS,
    M1,
    PRECISION,
    VARIANTS,
    MetalTcError,
)
from test_kernel_metal_runtime import importer, requires_metal

RAGGED = MatmulShape(45, 70, 37)


@functools.cache
def tc_runtime():
    """Compile both libraries once per test process; call only under requires_metal."""

    return MetalTcRuntime()


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
