"""Seeded inputs, pure-Python reference outputs and exact comparisons.

Inputs come from a SHA-256 counter stream instead of ``random``, so one seed
gives the same values on every platform and Python version. The references
compute every output element with plain loops and exact arithmetic, without
the simulator. Comparisons are exact: float32 values by bit pattern and
uint8 values by value.

This module also states the grayscale rule that every kernel must match:

* Weights: ``0.2989 R + 0.5870 G + 0.1140 B``, read as exact decimals.
* Rounding: the exact weighted sum rounds half up, ``floor(sum + 1/2)``.
* Clamping: the result is clamped to [0, 255]. The weights add up to 0.9999,
  so the largest sum is 254.9745 and the clamp never changes a uint8 input.

Binary floating point cannot hold 0.2989, 0.5870 or 0.1140 exactly, so a
float formula can put an exact .5 tie on either side. The exact rule cannot.
"""

from __future__ import annotations

import math
import re
import struct
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256

from .kernel_launch_buffer import DTYPES, encode_values
from .kernel_launch_config import KernelLaunchError, ceil_div, require_int

GRAYSCALE_WEIGHTS = ("0.2989", "0.5870", "0.1140")
GRAYSCALE_WEIGHTS_SOURCE = (
    "GPU MODE Lecture 3, Getting Started With CUDA for Python Programmers: its "
    "rgb-to-grayscale code weights R, G and B by 0.2989, 0.5870 and 0.1140. "
    "Each weight differs from the Rec. ITU-R BT.601 luma weight (0.299, 0.587, "
    "0.114) by at most 0.0001."
)
GRAYSCALE_RULE = (
    "Round the exact weighted sum half up, floor(sum + 1/2), then clamp the "
    "result to [0, 255]."
)
INPUT_LABEL = "ai-journey-kernel-launch-input"
MAX_INPUT_SEED = 2**32 - 1
MAX_INPUT_ELEMENTS = 1 << 22
_STREAM = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?")
_HALF = Fraction(1, 2)
_WEIGHTS = tuple(Fraction(weight) for weight in GRAYSCALE_WEIGHTS)


def seeded_bytes(seed: int, stream: str, count: int) -> bytes:
    """Return ``count`` bytes of one seeded stream of SHA-256 counter blocks.

    Block ``k`` is ``SHA-256(f"{INPUT_LABEL}:{stream}:{seed}:{k}")``. A stream
    name keeps two inputs with one seed independent, and a longer request
    starts with the bytes of a shorter one.
    """

    require_int(seed, "input seed", minimum=0, maximum=MAX_INPUT_SEED)
    if type(stream) is not str or _STREAM.fullmatch(stream) is None:
        raise KernelLaunchError("stream must be a lowercase slug")
    require_int(count, "count", minimum=1, maximum=4 * MAX_INPUT_ELEMENTS)
    blocks = [
        sha256(f"{INPUT_LABEL}:{stream}:{seed}:{counter}".encode("ascii")).digest()
        for counter in range(ceil_div(count, 32))
    ]
    return b"".join(blocks)[:count]


def seeded_uint8(seed: int, stream: str, count: int) -> tuple[int, ...]:
    """Return ``count`` uint8 values: the stream bytes themselves."""

    return tuple(seeded_bytes(seed, stream, count))


def seeded_float32(seed: int, stream: str, count: int) -> tuple[float, ...]:
    """Return float32 values of either sign with magnitudes in [2**-8, 2**8).

    Each value takes four stream bytes: one sign bit, a 4-bit exponent offset
    and a 23-bit mantissa. The spread of exponents makes many float32 sums
    round, so a bit-exact comparison tests the rounding as well as the
    indexing.
    """

    require_int(count, "count", minimum=1, maximum=MAX_INPUT_ELEMENTS)
    raw = seeded_bytes(seed, stream, 4 * count)
    values = []
    for offset in range(0, 4 * count, 4):
        word = int.from_bytes(raw[offset : offset + 4], "big")
        exponent = ((word >> 23) & 0xF) - 8
        magnitude = math.ldexp(0x800000 | (word & 0x7FFFFF), exponent - 23)
        values.append(-magnitude if word >> 31 else magnitude)
    return tuple(values)


def vector_add_inputs(seed: int, n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the seeded float32 operands ``a`` and ``b`` of one vector add."""

    return (
        seeded_float32(seed, "vector-add-a", n),
        seeded_float32(seed, "vector-add-b", n),
    )


def rgb_image(seed: int, width: int, height: int) -> tuple[int, ...]:
    """Return a seeded HxWx3 uint8 image, interleaved RGB in row-major order."""

    require_int(width, "width", minimum=1, maximum=MAX_INPUT_ELEMENTS)
    require_int(height, "height", minimum=1, maximum=MAX_INPUT_ELEMENTS)
    if width * height > MAX_INPUT_ELEMENTS:
        raise KernelLaunchError(f"images hold at most {MAX_INPUT_ELEMENTS} pixels")
    return seeded_uint8(seed, "rgb-image", 3 * width * height)


def _float32_values(values: Sequence[float], label: str) -> tuple[float, ...]:
    checked = tuple(values)
    if any(type(value) is not float or not math.isfinite(value) for value in checked):
        raise KernelLaunchError(f"{label} must hold finite floats")
    if tuple(array("f", checked)) != checked:
        raise KernelLaunchError(f"{label} must hold float32 values")
    return checked


def reference_vector_add(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    """Add float32 vectors element by element, one float32 rounding per sum.

    The sum of two float32 values is formed in double, then ``array('f')``
    rounds it to the nearest float32. For the seeded inputs the exponents
    differ by at most 15, so the double sum is exact and the result is the
    correctly rounded float32 sum.
    """

    left = _float32_values(a, "a")
    right = _float32_values(b, "b")
    if len(left) != len(right) or not left:
        raise KernelLaunchError("vector add operands need one equal, nonzero length")
    return tuple(array("f", [x + y for x, y in zip(left, right)]))


def rounded_sums(a: Sequence[float], b: Sequence[float]) -> int:
    """Count elements whose float32 sum differs from the exact sum."""

    exact = [Fraction(x) + Fraction(y) for x, y in zip(a, b)]
    return sum(
        Fraction(value) != total
        for value, total in zip(reference_vector_add(a, b), exact)
    )


def _image(rgb: Sequence[int], width: int, height: int) -> tuple[int, ...]:
    require_int(width, "width", minimum=1, maximum=MAX_INPUT_ELEMENTS)
    require_int(height, "height", minimum=1, maximum=MAX_INPUT_ELEMENTS)
    pixels = tuple(rgb)
    if len(pixels) != 3 * width * height:
        raise KernelLaunchError("an HxWx3 image needs 3 * width * height values")
    if any(type(value) is not int or not 0 <= value <= 255 for value in pixels):
        raise KernelLaunchError("image values must be uint8 integers")
    return pixels


def _exact_luma(red: int, green: int, blue: int) -> Fraction:
    return _WEIGHTS[0] * red + _WEIGHTS[1] * green + _WEIGHTS[2] * blue


def reference_rgb_to_grayscale(
    rgb: Sequence[int], width: int, height: int
) -> tuple[int, ...]:
    """Convert an HxWx3 uint8 image with the exact grayscale rule."""

    pixels = _image(rgb, width, height)
    gray = []
    for start in range(0, len(pixels), 3):
        exact = _exact_luma(*pixels[start : start + 3])
        gray.append(min(255, max(0, math.floor(exact + _HALF))))
    return tuple(gray)


def grayscale_rounding(rgb: Sequence[int], width: int, height: int) -> dict[str, int]:
    """Count pixels whose value depends on the rounding rule.

    ``ties`` are pixels whose exact sum ends in exactly .5, which the rule
    rounds up. ``truncation_differs`` counts pixels where truncation toward
    zero, the C conversion from float to ``unsigned char``, gives a value one
    lower than the rule.
    """

    pixels = _image(rgb, width, height)
    ties = truncated = 0
    for start in range(0, len(pixels), 3):
        exact = _exact_luma(*pixels[start : start + 3])
        fraction = exact - math.floor(exact)
        ties += fraction == _HALF
        truncated += fraction >= _HALF
    return {"ties": ties, "truncation_differs": truncated}


@dataclass(frozen=True)
class ExactComparison:
    """Element-by-element equality of two typed arrays, floats by bit pattern."""

    dtype: str
    elements: int
    mismatches: int
    first_mismatch: int | None

    @property
    def equal(self) -> bool:
        return self.mismatches == 0

    def to_mapping(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "elements": self.elements,
            "mismatches": self.mismatches,
            "first_mismatch": self.first_mismatch,
            "equal": self.equal,
        }


def compare_exact(
    dtype: str, expected: Sequence[int | float], actual: Sequence[int | float]
) -> ExactComparison:
    """Compare two arrays of ``dtype`` exactly; -0.0 and 0.0 differ as float32.

    Both arrays must hold valid values of ``dtype`` and have one length, so a
    shape or type error never reads as a mismatch count.
    """

    if dtype not in DTYPES:
        raise KernelLaunchError(f"dtype must be one of {DTYPES}")
    left, right = tuple(expected), tuple(actual)
    if len(left) != len(right):
        raise KernelLaunchError("compared arrays differ in length")
    encode_values(dtype, left)
    encode_values(dtype, right)
    if dtype == "float32":
        mismatched = [
            index
            for index, (x, y) in enumerate(zip(left, right))
            if struct.pack("<f", x) != struct.pack("<f", y)
        ]
    else:
        mismatched = [index for index, (x, y) in enumerate(zip(left, right)) if x != y]
    return ExactComparison(
        dtype=dtype,
        elements=len(left),
        mismatches=len(mismatched),
        first_mismatch=mismatched[0] if mismatched else None,
    )
