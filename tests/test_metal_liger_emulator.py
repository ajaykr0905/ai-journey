"""A NumPy float32 stand-in for the Day 80 runtime, for the CPU tests.

The stand-in has the methods of ``MetalLigerRuntime`` and computes each
kernel in plain float32 NumPy. It is not the GPU and makes no measurement,
so every report that it helps to make verifies only as synthetic timing.
"""

import functools

import numpy

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.metal_liger_plan import TORCH_SIDES
from ai_journey.metal_liger_reference import (
    dense_layout,
    liger_inputs,
    liger_reference,
    log64,
    sentinel_output,
)
from ai_journey.metal_liger_source import (
    FAMILIES,
    FAMILY_KERNELS,
    KERNELS,
    MetalLigerError,
    liger_library,
)
from test_metal_liger_reference import float32_kernel


class LigerEmulator:
    """The runtime's interface over float32 NumPy, with chosen pipeline limits."""

    backend = "injected"

    def __init__(self, limits=None, corrupt=None):
        self.library_source = liger_library()
        self.torch_version = "2.15.0"
        self.mps_switches = {
            "PYTORCH_MPS_FAST_MATH": None,
            "PYTORCH_MPS_PREFER_METAL": None,
        }
        self.limits = {
            name: PipelineLimits(1024, 32, kernel.threadgroup_bytes)
            for name, kernel in KERNELS.items()
        }
        self.limits.update(limits or {})
        self.corrupt = corrupt or (lambda family, output: output)
        self.synchronized = 0

    def synchronize(self):
        self.synchronized += 1

    def run(self, inputs, required):
        if inputs.layout.group_size(inputs.family) > required:
            raise MetalLigerError("a launch needs more threads than the run admitted")
        output = sentinel_output(inputs.family, inputs.layout)
        output[inputs.layout.output_valid(inputs.family)] = float32_kernel(inputs)
        return self.corrupt(inputs.family, output)

    def probe(self, function, arguments):
        values = arguments.astype(numpy.float64)
        if function == "exp":
            with numpy.errstate(under="ignore"):
                return numpy.exp(arguments)
        if function == "rsqrt":
            return (1.0 / numpy.sqrt(values)).astype(numpy.float32)
        return log64(values).astype(numpy.float32)

    def workload(self, inputs):
        if inputs.layout.gap or not inputs.layout.dense:
            raise MetalLigerError("a timing point packs dense rows with no gap")
        output = float32_kernel(inputs)
        kernel, side = FAMILY_KERNELS[inputs.family], TORCH_SIDES[inputs.family]
        return _Workload(
            {kernel: _noop, side: _noop},
            self.synchronize,
            lambda: {kernel: output.copy(), side: output.copy()},
        )


def _noop():
    return None


class _Workload:
    def __init__(self, sides, synchronize, outputs):
        self.sides = sides
        self.synchronize = synchronize
        self.outputs = outputs

    @property
    def calls(self):
        return tuple(self.sides.values())


@functools.cache
def ticking_clock(step=1000):
    """A clock that advances by ``step`` ns at each reading; synthetic only."""

    state = {"now": 0}

    def clock():
        state["now"] += step
        return state["now"]

    return clock


def test_the_stand_in_keeps_every_gap_and_meets_the_declared_bounds():
    emulator = LigerEmulator()
    for family in FAMILIES:
        layout = dense_layout(3, 40, 32)
        inputs = liger_inputs(80, family, 0, 0, layout, "uniform_80")
        output = emulator.run(inputs, 1024)
        valid = layout.output_valid(family)
        reference = liger_reference(inputs)
        error = numpy.abs(output[valid].astype(numpy.float64) - reference.output)
        assert numpy.all(error <= reference.bound)
        gaps = numpy.ones(output.size, dtype=bool)
        gaps[valid] = False
        assert numpy.all(numpy.isnan(output[gaps]))
