"""Seeded matmul inputs and an exact float32 reference in one stated order.

The operands come from the Day 73 SHA-256 counter streams through
``seeded_float32``, so one seed gives the same values on every platform. ``A``
is ``m x k`` and ``B`` is ``k x n``, both row-major float32, and the reference
computes each element of ``C = A @ B`` in the order that the kernels and the
pinned CUDA text use:

* the sum starts at +0.0;
* for ``i = 0, 1, ..., k - 1`` in turn, ``A[row, i] * B[i, col]`` rounds to
  float32, then the running sum plus that product rounds to float32.

Each rounding goes to the nearest float32, ties to even, as CUDA's
``__fmul_rn`` and ``__fadd_rn`` do. A fused multiply-add rounds once per step,
and another order rounds other partial sums, so either can change the bits:
the order is part of the rule. Python forms each product and each sum in
double, and ``to_float32`` rounds it. A product of two float32 values is exact
in double. Double has 53 bits, at least ``2 * 24 + 2``, so rounding a sum first
to double and then to float32 gives the correctly rounded float32 sum.

``correctly_rounded_matmul`` instead rounds each exact dot product once, with
integer arithmetic and no double step, to show how far the stated order moves
each element.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from .kernel_launch_buffer import encode_values, to_float32
from .kernel_launch_reference import seeded_float32
from .memory_access_tiling import MatmulShape

INPUT_STREAMS = ("matmul-a", "matmul-b")
SUMMATION_RULE = (
    "C[row, col] starts at +0.0. For i = 0, 1, ..., k - 1 in turn, the product "
    "A[row, i] * B[i, col] rounds to float32, then the running sum plus the "
    "product rounds to float32. Each rounding goes to the nearest float32, "
    "ties to even."
)
_FLOAT32_SIGNIFICAND_BITS = 24
_FLOAT32_MIN_EXPONENT = -126
_FLOAT32_MAX_EXPONENT = 127


class MatmulModelError(ValueError):
    """Raised when a matmul model input, run or report breaks its contract."""


def require_shape(shape: object) -> MatmulShape:
    """Return ``shape`` if it is the memory_access lab's ``MatmulShape``."""

    if type(shape) is not MatmulShape:
        raise TypeError("shape must be MatmulShape")
    return shape


def matmul_inputs(
    seed: int, shape: MatmulShape
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the seeded row-major float32 operands ``A`` and ``B`` of a shape.

    Each operand has its own stream, so a change to ``n`` leaves ``A`` alone.
    """

    checked = require_shape(shape)
    return (
        seeded_float32(seed, INPUT_STREAMS[0], checked.m * checked.k),
        seeded_float32(seed, INPUT_STREAMS[1], checked.k * checked.n),
    )


def _operands(
    a: Sequence[float], b: Sequence[float], shape: MatmulShape
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    checked = require_shape(shape)
    left, right = tuple(a), tuple(b)
    if len(left) != checked.m * checked.k or len(right) != checked.k * checked.n:
        raise MatmulModelError("A needs m * k values and B needs k * n values")
    # encode_values refuses any value that is not a finite float32.
    encode_values("float32", left)
    encode_values("float32", right)
    return left, right


def reference_matmul(
    a: Sequence[float], b: Sequence[float], shape: MatmulShape
) -> tuple[float, ...]:
    """Compute ``C = A @ B`` with plain loops in the stated order and roundings."""

    left, right = _operands(a, b, shape)
    m, n, k = shape.m, shape.n, shape.k
    out = []
    for row in range(m):
        start = row * k
        for col in range(n):
            total = 0.0
            for i in range(k):
                product = to_float32(left[start + i] * right[i * n + col])
                total = to_float32(total + product)
            out.append(total)
    return tuple(out)


def round_to_float32(value: Fraction) -> float:
    """Round an exact rational to the nearest float32, ties to even.

    Integer arithmetic finds the 24-bit significand, so no double rounding can
    occur. Values outside the normal float32 range are refused instead of
    being flushed or saturated.
    """

    if type(value) is not Fraction:
        raise TypeError("value must be a Fraction")
    if value == 0:
        return 0.0
    numerator, denominator = abs(value.numerator), value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        below = numerator < denominator << exponent
    else:
        below = numerator << -exponent < denominator
    exponent -= below
    shift = _FLOAT32_SIGNIFICAND_BITS - 1 - exponent
    if shift >= 0:
        quotient, remainder = divmod(numerator << shift, denominator)
        scale = denominator
    else:
        scale = denominator << -shift
        quotient, remainder = divmod(numerator, scale)
    if 2 * remainder > scale or (2 * remainder == scale and quotient & 1):
        quotient += 1
    magnitude = math.ldexp(quotient, -shift)
    if not 2.0**_FLOAT32_MIN_EXPONENT <= magnitude < 2.0 ** (_FLOAT32_MAX_EXPONENT + 1):
        raise MatmulModelError(f"{value} is outside the normal float32 range")
    return -magnitude if value < 0 else magnitude


def correctly_rounded_matmul(
    a: Sequence[float], b: Sequence[float], shape: MatmulShape
) -> tuple[float, ...]:
    """Round each exact dot product of ``A @ B`` once to float32."""

    left, right = _operands(a, b, shape)
    m, n, k = shape.m, shape.n, shape.k
    exact_a = [Fraction(value) for value in left]
    exact_b = [Fraction(value) for value in right]
    return tuple(
        round_to_float32(
            sum(
                (exact_a[row * k + i] * exact_b[i * n + col] for i in range(k)),
                Fraction(0),
            )
        )
        for row in range(m)
        for col in range(n)
    )


def _ordered_bits(value: float) -> int:
    bits = struct.unpack("<i", struct.pack("<f", value))[0]
    return bits if bits >= 0 else -(bits & 0x7FFFFFFF)


def ulp_distance(first: float, second: float) -> int:
    """Count the float32 values from ``first`` to ``second``; -0.0 equals 0.0."""

    encode_values("float32", (first, second))
    return abs(_ordered_bits(first) - _ordered_bits(second))


@dataclass(frozen=True)
class OrderRounding:
    """How the stated order compares with one rounding of each exact dot."""

    elements: int
    differing: int
    max_ulp_difference: int

    def to_mapping(self) -> dict[str, int]:
        return {
            "elements": self.elements,
            "differing": self.differing,
            "max_ulp_difference": self.max_ulp_difference,
        }


def order_rounding(
    stated: Sequence[float], correctly_rounded: Sequence[float]
) -> OrderRounding:
    """Count elements where the stated order differs from the exact result."""

    left, right = tuple(stated), tuple(correctly_rounded)
    if len(left) != len(right) or not left:
        raise MatmulModelError("compared results need one equal, nonzero length")
    distances = [ulp_distance(x, y) for x, y in zip(left, right)]
    return OrderRounding(
        elements=len(left),
        differing=sum(distance > 0 for distance in distances),
        max_ulp_difference=max(distances),
    )
