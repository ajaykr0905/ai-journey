"""Observations recorded on the development host's Apple GPU, replayed on any host.

``RecordedBackend`` stands in for ``MetalRuntime`` where MPS is missing: it
computes outputs with NumPy and replays what the GPU produced on 2026-11-16
(torch 2.15.0, macOS 27.2, arm64). Reports built on it are synthetic, never
measurements.
"""

import numpy

from ai_journey.kernel_metal_geometry import CheckedLaunch, PipelineLimits, image_launch
from ai_journey.kernel_metal_reference import (
    ALL_TRIPLES_SHAPE,
    UINT8_SENTINEL,
    all_rgb_triples,
    array_sha256,
    float32_sentinels,
    rgb_to_gray_reference,
    uint8_sentinels,
    vector_add_reference,
)
from ai_journey.kernel_metal_runtime import DeviceWorkload
from ai_journey.kernel_metal_source import KERNELS, PROBE_FUSED_RESULT

# Both pipelines reported these limits.
RECORDED_LIMITS = PipelineLimits(
    max_threads_per_threadgroup=1024,
    thread_execution_width=32,
    static_threadgroup_memory_bytes=0,
)
RECORDED_TORCH_VERSION = "2.15.0"
# The contraction probe returned 2**-24: the compiler fused the multiply-add.
RECORDED_CONTRACTION = PROBE_FUSED_RESULT
# (flat pixel index, Metal level) for every pixel of the all-triples image
# where the Metal kernel differed from the float32 reference. Three runs and
# 16 x 16 and 32 x 8 threadgroups gave the same 21 pixels.
RECORDED_ALL_TRIPLES_MISMATCHES = (
    (669970, 38),
    (670691, 63),
    (671633, 56),
    (683937, 86),
    (1341132, 98),
    (1350831, 117),
    (1367624, 144),
    (2023877, 163),
    (7921339, 187),
    (11164471, 111),
    (15101710, 136),
    (15757963, 155),
    (15767662, 174),
    (15777361, 193),
    (15787060, 212),
    (15791328, 241),
    (16400249, 126),
    (16412553, 156),
    (16433614, 212),
    (16441650, 213),
    (16443313, 231),
)
ALL_TRIPLES_SHA256 = "95eeb80877c99cdcb38755b9bb5ed29066bf70e870ea6eff9ee30285bd4cd5b7"


class RecordedBackend:
    """A NumPy stand-in for ``MetalRuntime`` that replays recorded GPU output."""

    backend = "recorded"
    torch_version = RECORDED_TORCH_VERSION

    def __init__(self, mismatches=RECORDED_ALL_TRIPLES_MISMATCHES, limits=None):
        self.mismatches = mismatches
        self.limits = {name: limits or RECORDED_LIMITS for name in KERNELS}
        self.launches = []

    def synchronize(self):
        pass

    def contraction_probe(self):
        return RECORDED_CONTRACTION

    def vector_add(self, first, second, launch):
        self.launches.append(launch)
        tail = float32_sentinels(launch.geometry.padding_threads)
        return numpy.concatenate([vector_add_reference(first, second), tail])

    def rgb_to_gray(self, rgb, launch):
        self.launches.append(launch)
        gray = rgb_to_gray_reference(rgb).reshape(-1)
        if rgb.shape == (*ALL_TRIPLES_SHAPE, 3) and array_sha256(rgb) == (
            ALL_TRIPLES_SHA256
        ):
            for index, level in self.mismatches:
                gray[index] = level
        tail = uint8_sentinels(launch.geometry.padding_threads)
        return numpy.concatenate([gray, tail])

    def torch_vector_add(self, first, second, device):
        return vector_add_reference(first, second)

    def torch_rgb_to_gray(self, rgb, device):
        return rgb_to_gray_reference(rgb)

    def vector_add_workload(self, first, second, launch):
        self.launches.append(launch)
        return DeviceWorkload(
            metal=lambda: numpy.add(first, second),
            torch=lambda: numpy.add(first, second),
            synchronize=self.synchronize,
            check=lambda: True,
        )

    def rgb_to_gray_workload(self, rgb, launch):
        self.launches.append(launch)
        return DeviceWorkload(
            metal=lambda: rgb_to_gray_reference(rgb),
            torch=lambda: rgb_to_gray_reference(rgb),
            synchronize=self.synchronize,
            check=lambda: True,
        )


def test_every_recorded_mismatch_is_a_one_level_flip_at_an_exact_half():
    image = all_rgb_triples()
    reference = rgb_to_gray_reference(image).reshape(-1)
    pixels = image.reshape(-1, 3).astype(numpy.int64)

    assert array_sha256(image) == ALL_TRIPLES_SHA256
    for index, level in RECORDED_ALL_TRIPLES_MISMATCHES:
        red, green, blue = pixels[index]
        # The exact decimal value is k + 0.5, a tie that a fused multiply-add
        # and two float32 roundings can settle on different sides.
        assert (2989 * red + 5870 * green + 1140 * blue) % 10000 == 5000
        assert abs(level - int(reference[index])) == 1


def test_the_recorded_backend_replays_its_mismatches_only_on_all_triples():
    backend = RecordedBackend()
    image = all_rgb_triples()
    launch = CheckedLaunch(image_launch(4096, 4096, (48, 16)), RECORDED_LIMITS)
    reference = rgb_to_gray_reference(image).reshape(-1)

    buffer = backend.rgb_to_gray(image, launch)

    changed = numpy.nonzero(buffer[: reference.size] != reference)[0]
    assert changed.tolist() == [index for index, _ in RECORDED_ALL_TRIPLES_MISMATCHES]
    assert launch.geometry.padding_threads == 32 * 4096
    assert set(buffer[reference.size :].tolist()) == {UINT8_SENTINEL}
    stripe = image[:16].copy()
    launch = CheckedLaunch(image_launch(16, 4096, (16, 16)), RECORDED_LIMITS)
    assert numpy.array_equal(
        backend.rgb_to_gray(stripe, launch), rgb_to_gray_reference(stripe).reshape(-1)
    )
