"""A NumPy float32 stand-in for the Day 81 runtime, for the CPU tests.

The stand-in has the methods of ``MetalFlashRuntime`` and computes each
side in plain float32 NumPy. It is not the GPU and makes no measurement,
so every report that it helps to make verifies only as synthetic timing.
"""

import numpy
import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.metal_flash_plan import SIDES
from ai_journey.metal_flash_reference import (
    dense_layout,
    flash_inputs,
    flash_reference,
    gather_rows,
    sentinel_output,
)
from ai_journey.metal_flash_runtime import FlashWorkload
from ai_journey.metal_flash_source import (
    CAUSAL,
    FULL,
    HEAD_DIM,
    KERNELS,
    MODES,
    N0,
    MetalFlashError,
    flash_library,
)
from ai_journey.metal_fused_reference import exp64


def float32_attention(inputs, mode):
    """Return a flat float32 O buffer: gap rows keep the sentinel."""

    layout = inputs.layout
    output = sentinel_output(layout).reshape(-1, HEAD_DIM)
    for index, start in enumerate(layout.starts()):
        q, k, v = inputs.sequence(index)
        scores = (q @ k.T) * numpy.float32(0.125)
        if mode == CAUSAL:
            visible = numpy.tril(numpy.ones(scores.shape, dtype=bool))
            scores = numpy.where(visible, scores, -numpy.inf).astype(numpy.float32)
        weights = numpy.exp(scores - scores.max(axis=1, keepdims=True))
        weights = weights / weights.sum(axis=1, keepdims=True)
        output[start : start + q.shape[0]] = (weights @ v).astype(numpy.float32)
    return output.reshape(-1)


def readings(current=1 << 20, driver=1 << 21):
    counters = {"current_bytes": current, "driver_bytes": driver}
    return {
        "before": dict(counters),
        "after_allocation": dict(counters),
        "after_call": dict(counters),
        "peak_during_call": dict(counters),
        "polls": 3,
    }


class FlashEmulator:
    """The runtime's interface over float32 NumPy, with chosen pipeline limits."""

    backend = "injected"

    def __init__(self, limits=None, corrupt=None, memory=None):
        self.library_source = flash_library()
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
        self.corrupt = corrupt or (lambda side, output: output)
        self.memory_readings = memory or (lambda side: readings())
        self.synchronized = 0

    def synchronize(self):
        self.synchronized += 1

    def run(self, inputs, variant, mode):
        return self.corrupt(variant, float32_attention(inputs, mode))

    def exp_probe(self, arguments):
        return exp64(arguments.astype(numpy.float64)).astype(numpy.float32)

    def memory(self, inputs, mode):
        return {side: self.memory_readings(side) for side in SIDES}

    def workload(self, inputs, mode):
        layout = inputs.layout
        if layout.gap != 0 or not layout.dense:
            raise MetalFlashError("a timing point packs dense sequences with no gap")
        result = float32_attention(inputs, mode)
        calls = {side: (lambda: None) for side in SIDES}

        def outputs():
            return {side: self.corrupt(side, result.copy()) for side in SIDES}

        return FlashWorkload(sides=calls, synchronize=self.synchronize, outputs=outputs)


@pytest.mark.parametrize("mode", MODES)
def test_the_emulator_meets_the_declared_bound(mode):
    inputs = flash_inputs(81, 1, 0, dense_layout(2, 70), "wide_scores")
    reference = flash_reference(inputs, mode)
    output = FlashEmulator().run(inputs, N0, mode)

    error = numpy.abs(gather_rows(output, reference.rows) - reference.output)
    assert numpy.max(error / reference.bound) <= 1.0
    assert numpy.max(error / reference.vendor_bound) <= 1.0


def test_the_emulator_refuses_a_gapped_timing_point():
    inputs = flash_inputs(81, 1, 0, dense_layout(2, 8, gap=1), "normal")

    with pytest.raises(MetalFlashError, match="dense sequences"):
        FlashEmulator().workload(inputs, FULL)
