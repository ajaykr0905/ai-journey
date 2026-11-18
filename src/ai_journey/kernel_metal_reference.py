"""Seeded kernel inputs and NumPy CPU references, compared through SHA-256 digests.

Every case draws from its own stream, ``numpy.random.default_rng([seed,
kernel, case])``, so a run and an independent verifier rebuild identical
inputs from the seed alone. Vector add sums two standard-normal float32
vectors. Grayscale converts uniform uint8 RGB pixels, or every one of the
2**24 RGB triples once in a 4096 x 4096 image; images are row-major with the
three channels last.

Grayscale follows the exact rule of the CUDA half of the lab. The weights
0.2989, 0.5870 and 0.1140 are exact decimals, so ``2989 R + 5870 G + 1140 B``
is the weighted sum times 10000 as an integer. Adding 5000 and dividing by
10000 with floor division rounds that sum half up, and the result is clamped
to 255. The weights sum to 0.9999, so the clamp never changes a value; it
stays as a guard. Binary floating point cannot hold the decimal weights, so
a float32 form of the rule rounds 19 of the 1703 exact ties down, and a
compiler that fuses a multiply and an add moves other ties. Integer
arithmetic has no rounding to move, so the kernel, PyTorch's equivalent and
this reference all use it, and every output must match the reference
exactly: ``MAX_LEVEL_DIFFERENCE`` is 0. Comparisons still list every
mismatched pixel with its value, so a failed check can say how far off an
output is.

A digest hashes the C-order bytes of an array (little-endian float32 or
uint8), so equal digests mean bit-identical arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy

from .kernel_metal_geometry import (
    MAX_THREADGROUP_THREADS,
    KernelMetalError,
    require_integer,
)

VECTOR_ADD_STREAM = 0
RGB_TO_GRAY_STREAM = 1
MAX_SEED = 2**32 - 1
MAX_CASE_INDEX = 63
MAX_VECTOR_LENGTH = 1 << 25
# A guarded launch of n elements in groups of g spans ceil(n / g) * g < n + g
# threads, so its output buffer can hold up to g - 1 elements past the data.
MAX_VECTOR_BUFFER = MAX_VECTOR_LENGTH + MAX_THREADGROUP_THREADS - 1
MAX_IMAGE_SIDE = 8192
MAX_IMAGE_PIXELS = 1 << 24
ALL_TRIPLES_SHAPE = (4096, 4096)
GRAY_WEIGHTS = (0.2989, 0.5870, 0.1140)
# The same weights as exact integers, as the CUDA half names them: the
# weighted sum times GRAY_SCALE.
GRAY_SCALE = 10000
GRAY_WEIGHTS_PER_10000 = (2989, 5870, 1140)
MAX_LEVEL = 255
MAX_LEVEL_DIFFERENCE = 0
MAX_RECORDED_MISMATCHES = 1024
# Sentinels prefill device output buffers; a guard tail must keep them.
FLOAT32_SENTINEL_BITS = 0x7FC00000
UINT8_SENTINEL = 0xA5
CHUNK_PIXELS = 1 << 20
_LITTLE_FLOAT32 = numpy.dtype("<f4")


def case_generator(seed: int, stream: int, index: int) -> numpy.random.Generator:
    """Return the generator of one case, independent of every other case."""

    require_integer(seed, "seed", 0, MAX_SEED)
    require_integer(stream, "stream", VECTOR_ADD_STREAM, RGB_TO_GRAY_STREAM)
    require_integer(index, "case index", 0, MAX_CASE_INDEX)
    return numpy.random.default_rng([seed, stream, index])


def vector_add_inputs(
    seed: int, index: int, length: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Draw the two standard-normal float32 operands of one vector-add case."""

    count = require_integer(length, "length", 1, MAX_VECTOR_LENGTH)
    generator = case_generator(seed, VECTOR_ADD_STREAM, index)
    first = generator.standard_normal(count, dtype=numpy.float32)
    second = generator.standard_normal(count, dtype=numpy.float32)
    return first, second


def require_image_shape(height: int, width: int) -> tuple[int, int]:
    """Bound an image so its pixel count and flat RGB index fit the kernels."""

    rows = require_integer(height, "height", 1, MAX_IMAGE_SIDE)
    columns = require_integer(width, "width", 1, MAX_IMAGE_SIDE)
    if rows * columns > MAX_IMAGE_PIXELS:
        raise KernelMetalError(f"an image may hold at most {MAX_IMAGE_PIXELS} pixels")
    return rows, columns


def uniform_rgb_image(seed: int, index: int, height: int, width: int) -> numpy.ndarray:
    """Draw one image of uniform uint8 RGB pixels, shape ``(height, width, 3)``."""

    shape = require_image_shape(height, width)
    generator = case_generator(seed, RGB_TO_GRAY_STREAM, index)
    return generator.integers(0, 256, size=(*shape, 3), dtype=numpy.uint8)


def all_rgb_triples() -> numpy.ndarray:
    """Return every RGB triple once: pixel ``p`` holds ``(p >> 16, p >> 8, p)``.

    The 4096 x 4096 image covers the whole input domain of the grayscale
    kernel, so its comparison finds every pixel value that can differ.
    """

    codes = numpy.arange(1 << 24, dtype=numpy.uint32)
    image = numpy.empty((codes.size, 3), dtype=numpy.uint8)
    image[:, 0] = codes >> 16
    image[:, 1] = (codes >> 8) & 0xFF
    image[:, 2] = codes & 0xFF
    return image.reshape(*ALL_TRIPLES_SHAPE, 3)


def _vector(values: object, label: str) -> numpy.ndarray:
    if (
        not isinstance(values, numpy.ndarray)
        or values.dtype != numpy.float32
        or values.ndim != 1
        or not 1 <= values.size <= MAX_VECTOR_LENGTH
    ):
        raise KernelMetalError(f"{label} must be a 1D float32 array")
    return values


def _image(rgb: object) -> numpy.ndarray:
    if (
        not isinstance(rgb, numpy.ndarray)
        or rgb.dtype != numpy.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
    ):
        raise KernelMetalError("rgb must be a uint8 array of shape (height, width, 3)")
    require_image_shape(int(rgb.shape[0]), int(rgb.shape[1]))
    return rgb


def vector_add_reference(first: numpy.ndarray, second: numpy.ndarray) -> numpy.ndarray:
    """Add in float32 on the CPU; IEEE addition makes the answer unique."""

    left, right = _vector(first, "first"), _vector(second, "second")
    if left.shape != right.shape:
        raise KernelMetalError("vector-add operands must have the same length")
    return numpy.add(left, right)


def rgb_to_gray_reference(rgb: numpy.ndarray) -> numpy.ndarray:
    """Convert RGB to gray with the exact integer rule: round half up, clamp.

    The largest sum, ``9999 * 255 + 5000``, fits in 32 bits, and chunks of
    ``CHUNK_PIXELS`` keep the uint32 temporaries small next to the image.
    """

    image = _image(rgb)
    height, width = int(image.shape[0]), int(image.shape[1])
    flat = numpy.ascontiguousarray(image).reshape(-1, 3)
    gray = numpy.empty(height * width, dtype=numpy.uint8)
    red, green, blue = (numpy.uint32(weight) for weight in GRAY_WEIGHTS_PER_10000)
    half, scale = numpy.uint32(GRAY_SCALE // 2), numpy.uint32(GRAY_SCALE)
    for start in range(0, flat.shape[0], CHUNK_PIXELS):
        block = flat[start : start + CHUNK_PIXELS]
        level = red * block[:, 0].astype(numpy.uint32)
        level += green * block[:, 1].astype(numpy.uint32)
        level += blue * block[:, 2].astype(numpy.uint32)
        level += half
        level //= scale
        numpy.minimum(level, numpy.uint32(MAX_LEVEL), out=level)
        gray[start : start + block.shape[0]] = level.astype(numpy.uint8)
    return gray.reshape(height, width)


def array_sha256(array: numpy.ndarray) -> str:
    """Hash the C-order bytes of a float32 or uint8 array, little-endian."""

    if not isinstance(array, numpy.ndarray) or array.dtype not in (
        numpy.float32,
        numpy.uint8,
    ):
        raise KernelMetalError("only float32 and uint8 arrays are hashed")
    if array.dtype == numpy.float32:
        array = array.astype(_LITTLE_FLOAT32, copy=False)
    return sha256(numpy.ascontiguousarray(array).data).hexdigest()


def float32_sentinels(count: int) -> numpy.ndarray:
    """Return ``count`` quiet NaNs with fixed bits, so a hash can expect them.

    The bound is the longest guarded vector launch, padding threads included.
    """

    require_integer(count, "sentinel count", 0, MAX_VECTOR_BUFFER)
    return numpy.full(count, FLOAT32_SENTINEL_BITS, dtype=numpy.uint32).view(
        numpy.float32
    )


def uint8_sentinels(count: int) -> numpy.ndarray:
    """Return ``count`` copies of the uint8 sentinel byte."""

    require_integer(count, "sentinel count", 0, MAX_IMAGE_PIXELS * 3)
    return numpy.full(count, UINT8_SENTINEL, dtype=numpy.uint8)


@dataclass(frozen=True)
class LevelComparison:
    """Every pixel where a device output differs from the reference, by value.

    ``mismatches`` holds ``(flat pixel index, device level)`` pairs in index
    order, enough to rebuild the device output from the reference exactly.
    """

    pixels: int
    mismatches: tuple[tuple[int, int], ...]
    max_abs_difference: int

    def __post_init__(self) -> None:
        require_integer(self.pixels, "pixels", 1, MAX_IMAGE_PIXELS)
        if (
            type(self.mismatches) is not tuple
            or len(self.mismatches) > MAX_RECORDED_MISMATCHES
        ):
            raise KernelMetalError(
                f"at most {MAX_RECORDED_MISMATCHES} mismatched pixels are recorded"
            )
        previous = -1
        for item in self.mismatches:
            if type(item) is not tuple or len(item) != 2:
                raise KernelMetalError("a mismatch must be an (index, level) pair")
            index = require_integer(item[0], "mismatch index", 0, self.pixels - 1)
            require_integer(item[1], "mismatch level", 0, 255)
            if index <= previous:
                raise KernelMetalError("mismatch indices must be strictly increasing")
            previous = index
        require_integer(self.max_abs_difference, "max_abs_difference", 0, 255)
        if (self.max_abs_difference == 0) != (not self.mismatches):
            raise KernelMetalError("a nonzero difference needs a mismatched pixel")

    @property
    def mismatched_pixels(self) -> int:
        """Pixels whose device level differs from the reference level."""

        return len(self.mismatches)

    def to_mapping(self) -> dict[str, object]:
        """Return the comparison as a report records it."""

        return {
            "mismatches": [list(item) for item in self.mismatches],
            "mismatched_pixels": self.mismatched_pixels,
            "max_abs_difference_levels": self.max_abs_difference,
        }


def compare_levels(output: numpy.ndarray, reference: numpy.ndarray) -> LevelComparison:
    """Compare two gray images pixel by pixel and record every mismatch.

    More than ``MAX_RECORDED_MISMATCHES`` mismatches is an error, because a
    report could not rebuild the output from what it records.
    """

    for label, array in (("output", output), ("reference", reference)):
        if not isinstance(array, numpy.ndarray) or array.dtype != numpy.uint8:
            raise KernelMetalError(f"{label} must be a uint8 array")
    if output.shape != reference.shape:
        raise KernelMetalError("output and reference shapes differ")
    flat_output = numpy.ascontiguousarray(output).reshape(-1)
    flat_reference = numpy.ascontiguousarray(reference).reshape(-1)
    (indices,) = numpy.nonzero(flat_output != flat_reference)
    if indices.size > MAX_RECORDED_MISMATCHES:
        raise KernelMetalError(
            f"{indices.size} pixels differ from the reference; at most "
            f"{MAX_RECORDED_MISMATCHES} can be recorded"
        )
    levels = flat_output[indices]
    difference = numpy.abs(levels.astype(numpy.int16) - flat_reference[indices])
    return LevelComparison(
        pixels=int(flat_reference.size),
        mismatches=tuple(
            (int(index), int(level)) for index, level in zip(indices, levels)
        ),
        max_abs_difference=int(difference.max()) if indices.size else 0,
    )
