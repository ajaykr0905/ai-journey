"""Correctness of the Metal kernels against PyTorch on MPS and on the CPU.

Each case draws seeded inputs, computes the NumPy CPU reference, and runs
three implementations: the Metal kernel and PyTorch's equivalent on the MPS
device, and PyTorch's equivalent on the CPU. The Metal output buffer starts
as sentinels and is longer than the data by the launch's padding threads;
that guard tail must keep its sentinels, which shows the bounds guard
stopped every padding thread. Every output is recorded by the SHA-256 digest
of its whole buffer.

Vector add must match the reference bit for bit on all three. Grayscale
uses the exact integer rule, so its outputs must match too: a grayscale
output may differ by at most ``MAX_LEVEL_DIFFERENCE`` levels, which is 0.
Each grayscale record keeps the list of mismatched pixels with their values,
so a verifier rebuilds each output from the reference and checks its digest.
A case that breaks either rule stops the run, so a report never holds a
failed check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .kernel_metal_geometry import (
    CheckedLaunch,
    KernelMetalError,
    image_launch,
    require_integer,
    vector_launch,
)
from .kernel_metal_reference import (
    ALL_TRIPLES_SHAPE,
    FLOAT32_SENTINEL_BITS,
    MAX_CASE_INDEX,
    MAX_LEVEL_DIFFERENCE,
    UINT8_SENTINEL,
    LevelComparison,
    all_rgb_triples,
    array_sha256,
    compare_levels,
    rgb_to_gray_reference,
    uniform_rgb_image,
    vector_add_inputs,
    vector_add_reference,
)
from .kernel_metal_source import RGB_TO_GRAY, VECTOR_ADD

METAL_KERNEL = "metal_kernel"
TORCH = "torch"
# (implementation, device) in the order every case records them.
IMPLEMENTATIONS = ((METAL_KERNEL, "mps"), (TORCH, "mps"), (TORCH, "cpu"))
UNIFORM = "uniform"
ALL_RGB_TRIPLES = "all_rgb_triples"
PATTERNS = (UNIFORM, ALL_RGB_TRIPLES)


@dataclass(frozen=True)
class OutputRecord:
    """One implementation's output buffer: where it ran, its digest, its errors.

    ``comparison`` is ``None`` for vector add, whose outputs must be exact.
    """

    implementation: str
    device: str
    buffer_sha256: str
    guard_elements: int
    comparison: LevelComparison | None

    def to_mapping(self) -> dict[str, object]:
        """Return the record as a report stores it."""

        mapping: dict[str, object] = {
            "implementation": self.implementation,
            "device": self.device,
            "buffer_sha256": self.buffer_sha256,
            "guard_elements": self.guard_elements,
        }
        if self.comparison is not None:
            mapping.update(self.comparison.to_mapping())
        return mapping


@dataclass(frozen=True)
class VectorAddCase:
    """One seeded vector-add case, exact on every implementation."""

    index: int
    launch: CheckedLaunch
    a_sha256: str
    b_sha256: str
    reference_sha256: str
    outputs: tuple[OutputRecord, ...]

    @property
    def length(self) -> int:
        """Elements in each operand and in the result."""

        return self.launch.geometry.extent[0]

    def to_mapping(self) -> dict[str, object]:
        """Return the case as a report stores it."""

        return {
            "index": self.index,
            "length": self.length,
            "launch": self.launch.to_mapping(),
            "a_sha256": self.a_sha256,
            "b_sha256": self.b_sha256,
            "reference_sha256": self.reference_sha256,
            "outputs": [output.to_mapping() for output in self.outputs],
        }


@dataclass(frozen=True)
class GrayCase:
    """One grayscale case, every output within the level bound."""

    index: int
    pattern: str
    launch: CheckedLaunch
    rgb_sha256: str
    reference_sha256: str
    outputs: tuple[OutputRecord, ...]

    @property
    def shape(self) -> tuple[int, int]:
        """The image as ``(height, width)``."""

        width, height = self.launch.geometry.extent
        return height, width

    @property
    def max_abs_difference(self) -> int:
        """The largest level difference of any implementation in this case."""

        return max(output.comparison.max_abs_difference for output in self.outputs)

    @property
    def mismatched_pixels(self) -> dict[str, int]:
        """Mismatched pixels per ``implementation/device``."""

        return {
            f"{output.implementation}/{output.device}": (
                output.comparison.mismatched_pixels
            )
            for output in self.outputs
        }

    def to_mapping(self) -> dict[str, object]:
        """Return the case as a report stores it."""

        height, width = self.shape
        return {
            "index": self.index,
            "pattern": self.pattern,
            "height": height,
            "width": width,
            "launch": self.launch.to_mapping(),
            "rgb_sha256": self.rgb_sha256,
            "reference_sha256": self.reference_sha256,
            "bound_levels": MAX_LEVEL_DIFFERENCE,
            "outputs": [output.to_mapping() for output in self.outputs],
        }


def _flat_buffer(
    buffer: object, dtype: type, elements: int, label: str
) -> numpy.ndarray:
    if (
        not isinstance(buffer, numpy.ndarray)
        or buffer.dtype != dtype
        or buffer.shape != (elements,)
    ):
        raise KernelMetalError(
            f"{label}: the Metal kernel returned no {elements}-element buffer"
        )
    return buffer


def check_vector_add(
    backend: object, *, seed: int, index: int, length: int, group_size: int
) -> VectorAddCase:
    """Run one vector-add case on every implementation and require exactness."""

    require_integer(index, "case index", 0, MAX_CASE_INDEX)
    first, second = vector_add_inputs(seed, index, length)
    reference = vector_add_reference(first, second)
    launch = CheckedLaunch(
        vector_launch(length, group_size), backend.limits[VECTOR_ADD]
    )
    label = f"{VECTOR_ADD} case {index} ({length} elements)"
    guard = launch.geometry.padding_threads
    buffer = _flat_buffer(
        backend.vector_add(first, second, launch), numpy.float32, length + guard, label
    )
    if not numpy.array_equal(
        buffer[:length].view(numpy.uint32), reference.view(numpy.uint32)
    ):
        raise KernelMetalError(
            f"{label}: the Metal kernel differs from the CPU reference"
        )
    if numpy.any(buffer[length:].view(numpy.uint32) != FLOAT32_SENTINEL_BITS):
        raise KernelMetalError(f"{label}: a padding thread wrote into the guard tail")
    outputs = [OutputRecord(METAL_KERNEL, "mps", array_sha256(buffer), guard, None)]
    for implementation, device in IMPLEMENTATIONS[1:]:
        result = backend.torch_vector_add(first, second, device)
        if (
            not isinstance(result, numpy.ndarray)
            or result.dtype != numpy.float32
            or not numpy.array_equal(
                result.view(numpy.uint32), reference.view(numpy.uint32)
            )
        ):
            raise KernelMetalError(
                f"{label}: {implementation} on {device} differs from the CPU reference"
            )
        outputs.append(
            OutputRecord(implementation, device, array_sha256(result), 0, None)
        )
    return VectorAddCase(
        index=index,
        launch=launch,
        a_sha256=array_sha256(first),
        b_sha256=array_sha256(second),
        reference_sha256=array_sha256(reference),
        outputs=tuple(outputs),
    )


def _within_bound(
    output: numpy.ndarray, reference: numpy.ndarray, label: str
) -> LevelComparison:
    comparison = compare_levels(output, reference)
    if comparison.max_abs_difference > MAX_LEVEL_DIFFERENCE:
        raise KernelMetalError(
            f"{label} differs by {comparison.max_abs_difference} levels; the bound "
            f"is {MAX_LEVEL_DIFFERENCE}"
        )
    return comparison


def check_rgb_to_gray(
    backend: object,
    *,
    seed: int,
    index: int,
    shape: tuple[int, int],
    group_size: tuple[int, int],
    pattern: str = UNIFORM,
) -> GrayCase:
    """Run one grayscale case on every implementation within the level bound.

    ``all_rgb_triples`` replaces the seeded image with every RGB triple once,
    which needs the 4096 x 4096 shape.
    """

    require_integer(index, "case index", 0, MAX_CASE_INDEX)
    if type(shape) is not tuple or len(shape) != 2:
        raise KernelMetalError("an image shape must be a (height, width) tuple")
    height, width = shape
    if pattern == UNIFORM:
        image = uniform_rgb_image(seed, index, height, width)
    elif pattern == ALL_RGB_TRIPLES:
        if shape != ALL_TRIPLES_SHAPE:
            raise KernelMetalError("all_rgb_triples needs a 4096 x 4096 image")
        image = all_rgb_triples()
    else:
        raise KernelMetalError(f"pattern must be one of {', '.join(PATTERNS)}")
    reference = rgb_to_gray_reference(image)
    launch = CheckedLaunch(
        image_launch(height, width, group_size), backend.limits[RGB_TO_GRAY]
    )
    label = f"{RGB_TO_GRAY} case {index} ({height} x {width})"
    pixels, guard = height * width, launch.geometry.padding_threads
    buffer = _flat_buffer(
        backend.rgb_to_gray(image, launch), numpy.uint8, pixels + guard, label
    )
    if numpy.any(buffer[pixels:] != UINT8_SENTINEL):
        raise KernelMetalError(f"{label}: a padding thread wrote into the guard tail")
    comparison = _within_bound(
        buffer[:pixels].reshape(height, width), reference, f"{label}: the Metal kernel"
    )
    outputs = [
        OutputRecord(METAL_KERNEL, "mps", array_sha256(buffer), guard, comparison)
    ]
    for implementation, device in IMPLEMENTATIONS[1:]:
        result = backend.torch_rgb_to_gray(image, device)
        comparison = _within_bound(
            result, reference, f"{label}: {implementation} on {device}"
        )
        outputs.append(
            OutputRecord(implementation, device, array_sha256(result), 0, comparison)
        )
    return GrayCase(
        index=index,
        pattern=pattern,
        launch=launch,
        rgb_sha256=array_sha256(image),
        reference_sha256=array_sha256(reference),
        outputs=tuple(outputs),
    )
