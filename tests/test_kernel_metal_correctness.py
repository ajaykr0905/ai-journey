from hashlib import sha256

import numpy
import pytest

from ai_journey.kernel_metal_correctness import (
    ALL_RGB_TRIPLES,
    check_rgb_to_gray,
    check_vector_add,
)
from ai_journey.kernel_metal_geometry import KernelMetalError, PipelineLimits
from ai_journey.kernel_metal_reference import (
    all_rgb_triples,
    array_sha256,
    float32_sentinels,
    rgb_to_gray_reference,
    uint8_sentinels,
    uniform_rgb_image,
    vector_add_inputs,
    vector_add_reference,
)
from test_kernel_metal_recorded import RecordedBackend
from test_kernel_metal_runtime import metal_runtime, requires_metal


def test_a_vector_add_case_is_exact_and_hashes_the_guard_tail():
    case = check_vector_add(
        RecordedBackend(), seed=73, index=1, length=1000, group_size=256
    )
    first, second = vector_add_inputs(73, 1, 1000)
    reference = vector_add_reference(first, second)
    buffer = numpy.concatenate([reference, float32_sentinels(24)])

    mapping = case.to_mapping()
    assert mapping["length"] == 1000
    assert mapping["launch"]["padding_threads"] == 24
    assert mapping["a_sha256"] == array_sha256(first)
    assert mapping["b_sha256"] == array_sha256(second)
    assert mapping["reference_sha256"] == array_sha256(reference)
    assert mapping["outputs"] == [
        {
            "implementation": "metal_kernel",
            "device": "mps",
            "buffer_sha256": array_sha256(buffer),
            "guard_elements": 24,
        },
        {
            "implementation": "torch",
            "device": "mps",
            "buffer_sha256": array_sha256(reference),
            "guard_elements": 0,
        },
        {
            "implementation": "torch",
            "device": "cpu",
            "buffer_sha256": array_sha256(reference),
            "guard_elements": 0,
        },
    ]


def test_the_all_triples_case_is_exact_on_every_implementation():
    case = check_rgb_to_gray(
        RecordedBackend(),
        seed=73,
        index=0,
        shape=(4096, 4096),
        group_size=(16, 16),
        pattern=ALL_RGB_TRIPLES,
    )
    image = all_rgb_triples()
    reference = array_sha256(rgb_to_gray_reference(image))

    mapping = case.to_mapping()
    for output in mapping["outputs"]:
        assert output["buffer_sha256"] == reference
        assert (output["mismatched_pixels"], output["mismatches"]) == (0, [])
        assert output["max_abs_difference_levels"] == 0
    assert case.max_abs_difference == 0
    assert case.mismatched_pixels == {
        "metal_kernel/mps": 0,
        "torch/mps": 0,
        "torch/cpu": 0,
    }
    assert mapping["rgb_sha256"] == array_sha256(image)
    assert (mapping["pattern"], mapping["bound_levels"]) == (ALL_RGB_TRIPLES, 0)


def test_a_padded_image_case_keeps_its_tail_and_records_its_shape():
    case = check_rgb_to_gray(
        RecordedBackend(), seed=73, index=2, shape=(17, 33), group_size=(16, 16)
    )
    reference = rgb_to_gray_reference(uniform_rgb_image(73, 2, 17, 33))
    tail = uint8_sentinels(975)

    mapping = case.to_mapping()
    assert (mapping["height"], mapping["width"]) == (17, 33)
    assert mapping["pattern"] == "uniform"
    assert mapping["launch"]["threads"] == [48, 32]
    assert mapping["outputs"][0]["guard_elements"] == 975
    assert mapping["outputs"][0]["buffer_sha256"] == (
        sha256(reference.tobytes() + tail.tobytes()).hexdigest()
    )
    assert mapping["outputs"][1]["buffer_sha256"] == array_sha256(reference)
    assert case.shape == (17, 33)
    assert case.max_abs_difference == 0


class OffByOneUlp(RecordedBackend):
    def vector_add(self, first, second, launch):
        buffer = super().vector_add(first, second, launch)
        buffer[0] = numpy.nextafter(buffer[0], numpy.float32(numpy.inf))
        return buffer


class WritesPastTheGuard(RecordedBackend):
    def vector_add(self, first, second, launch):
        buffer = super().vector_add(first, second, launch)
        buffer[-1] = 0.0
        return buffer

    def rgb_to_gray(self, rgb, launch):
        buffer = super().rgb_to_gray(rgb, launch)
        buffer[-1] = 0
        return buffer


class TorchDiffers(RecordedBackend):
    def torch_vector_add(self, first, second, device):
        result = super().torch_vector_add(first, second, device)
        return result * numpy.float32(2) if device == "mps" else result


class ShortBuffer(RecordedBackend):
    def vector_add(self, first, second, launch):
        return super().vector_add(first, second, launch)[:-1]


class MetalOneLevelOff(RecordedBackend):
    """One pixel one level off: the exact rule leaves no tolerance for it."""

    def rgb_to_gray(self, rgb, launch):
        buffer = super().rgb_to_gray(rgb, launch)
        buffer[0] = (int(buffer[0]) + 1) % 256
        return buffer


class TorchOneLevelOff(RecordedBackend):
    def torch_rgb_to_gray(self, rgb, device):
        gray = super().torch_rgb_to_gray(rgb, device)
        if device == "cpu":
            gray[0, 0] = (int(gray[0, 0]) + 1) % 256
        return gray


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (OffByOneUlp(), "Metal kernel differs from the CPU reference"),
        (WritesPastTheGuard(), "wrote into the guard tail"),
        (TorchDiffers(), "torch on mps differs"),
        (ShortBuffer(), "returned no 1024-element buffer"),
        (RecordedBackend(limits=PipelineLimits(128, 32, 0)), "limit of 128"),
    ],
)
def test_vector_add_cases_stop_on_any_inexact_output(backend, message):
    with pytest.raises(KernelMetalError, match=message):
        check_vector_add(backend, seed=73, index=0, length=1000, group_size=256)


@pytest.mark.parametrize(
    ("backend", "shape", "pattern", "message"),
    [
        (
            MetalOneLevelOff(),
            (17, 33),
            "uniform",
            "the Metal kernel differs by 1 levels; the bound is 0",
        ),
        (
            TorchOneLevelOff(),
            (17, 33),
            "uniform",
            "torch on cpu differs by 1 levels; the bound is 0",
        ),
        (WritesPastTheGuard(), (17, 33), "uniform", "wrote into the guard tail"),
        (RecordedBackend(), (17, 33), ALL_RGB_TRIPLES, "4096 x 4096"),
        (RecordedBackend(), (17, 33), "gradient", "pattern must be one of"),
        (RecordedBackend(), [17, 33], "uniform", r"\(height, width\) tuple"),
    ],
)
def test_grayscale_cases_stop_beyond_the_bound_or_past_the_guard(
    backend, shape, pattern, message
):
    with pytest.raises(KernelMetalError, match=message):
        check_rgb_to_gray(
            backend,
            seed=73,
            index=0,
            shape=shape,
            group_size=(16, 16),
            pattern=pattern,
        )


@requires_metal
@pytest.mark.parametrize("length", [1, 255, 256, 257, 65537])
def test_metal_vector_add_is_exact_on_this_gpu(length):
    case = check_vector_add(
        metal_runtime(), seed=73, index=0, length=length, group_size=256
    )

    assert case.length == length
    assert case.outputs[1].buffer_sha256 == case.reference_sha256


@requires_metal
@pytest.mark.parametrize("shape", [(1, 1), (15, 17), (16, 16), (31, 1025)])
def test_metal_grayscale_is_exact_on_this_gpu(shape):
    case = check_rgb_to_gray(
        metal_runtime(), seed=73, index=0, shape=shape, group_size=(16, 16)
    )

    assert case.max_abs_difference == 0
    assert set(case.mismatched_pixels.values()) == {0}


@requires_metal
@pytest.mark.parametrize("group_size", [(16, 16), (32, 8), (8, 32)])
def test_metal_grayscale_is_exact_on_every_rgb_triple_in_any_group_shape(group_size):
    # Each shape orders the threads differently; the levels must not change.
    case = check_rgb_to_gray(
        metal_runtime(),
        seed=73,
        index=0,
        shape=(4096, 4096),
        group_size=group_size,
        pattern=ALL_RGB_TRIPLES,
    )

    assert case.max_abs_difference == 0
    assert set(case.mismatched_pixels.values()) == {0}
    exact = case.reference_sha256
    assert {output.buffer_sha256 for output in case.outputs} == {exact}
