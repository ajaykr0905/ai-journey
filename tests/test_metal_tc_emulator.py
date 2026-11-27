"""A NumPy float32 stand-in for the Day 82 runtime, for the CPU tests.

The stand-in has the methods of ``MetalTcRuntime`` and computes each side
with float32 products and sums in k order. It is not the GPU and makes no
measurement, so every report that it helps to make verifies only as
synthetic timing.
"""

import numpy
import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.metal_tc_plan import SIDE_PRECISION, SIDES, TORCH_BFLOAT16
from ai_journey.metal_tc_reference import (
    TAIL,
    operands,
    rounded,
    sentinel_output,
    tc_reference,
)
from ai_journey.metal_tc_runtime import COMPILE_FAILED, Support, TcWorkload
from ai_journey.metal_tc_source import (
    KERNELS,
    LIBRARIES,
    M3,
    PRECISION,
    VARIANTS,
    MetalTcError,
    library_source,
)


def float32_product(values, precision):
    """Return C in float32: each product and each sum rounded once, k by k."""

    a, b = values.inputs(precision)
    total = numpy.zeros((values.shape.m, values.shape.n), dtype=numpy.float32)
    for step in range(values.shape.k):
        total = total + numpy.multiply.outer(a[:, step], b[step, :])
    return total.reshape(-1)


class TcEmulator:
    """The runtime's interface over float32 NumPy, with chosen limits and support."""

    backend = "injected"

    def __init__(self, limits=None, corrupt=None, bfloat=True, torch_bfloat=True):
        self.library_sources = {name: library_source(name) for name in LIBRARIES}
        self.torch_version = "2.15.0"
        self.mps_switches = {
            "PYTORCH_MPS_FAST_MATH": None,
            "PYTORCH_MPS_PREFER_METAL": None,
        }
        self.limits = {
            name: PipelineLimits(1024, 32, kernel.threadgroup_bytes)
            for name, kernel in KERNELS.items()
            if bfloat or kernel.variant != M3
        }
        self.limits.update(limits or {})
        self.metal_version = 400
        self.support = {
            M3: Support(True, None) if bfloat else Support(False, COMPILE_FAILED),
            TORCH_BFLOAT16: Support(True, None)
            if torch_bfloat
            else Support(False, COMPILE_FAILED),
        }
        self.corrupt = corrupt or (lambda side, output: output)
        self.synchronized = 0

    def synchronize(self):
        self.synchronized += 1

    def supported(self, side):
        record = self.support.get(side)
        return record is None or record.supported

    def run(self, values, variant):
        if not self.supported(variant):
            raise MetalTcError(f"{variant} is not supported on this host")
        output = sentinel_output(values.shape)
        output[: values.shape.m * values.shape.n] = float32_product(
            values, PRECISION[variant]
        )
        return self.corrupt(variant, output)

    def workload(self, values):
        supported = tuple(side for side in SIDES if self.supported(side))
        results = {}
        for side in supported:
            product = float32_product(values, SIDE_PRECISION[side])
            if side not in VARIANTS:
                product = rounded(product, SIDE_PRECISION[side])
            results[side] = product
        calls = {side: (lambda: None) for side in SIDES}

        def outputs():
            return {side: self.corrupt(side, results[side].copy()) for side in results}

        return TcWorkload(
            sides=calls,
            supported=supported,
            synchronize=self.synchronize,
            outputs=outputs,
        )


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_emulator_meets_the_declared_bound_of_each_type(variant):
    values = operands(82, 0, 0, MatmulShape(9, 11, 300))
    reference = tc_reference(values)
    output = TcEmulator().run(values, variant)

    assert output.size == 9 * 11 + TAIL
    error = numpy.abs(output[:99].reshape(9, 11) - reference.output)
    assert numpy.max(error / reference.bound(PRECISION[variant])) <= 1.0


def test_the_emulator_drops_an_unsupported_side_from_its_outputs():
    emulator = TcEmulator(bfloat=False, torch_bfloat=False)
    workload = emulator.workload(operands(82, 1, 0, MatmulShape(8, 8, 8)))

    assert tuple(workload.sides) == SIDES
    assert M3 not in workload.supported and TORCH_BFLOAT16 not in workload.supported
    assert set(workload.outputs()) == set(workload.supported)
    with pytest.raises(MetalTcError):
        emulator.run(operands(82, 0, 0, MatmulShape(8, 8, 8)), M3)
