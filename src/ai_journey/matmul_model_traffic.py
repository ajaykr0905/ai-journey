"""FLOP and global-traffic accounting of the naive matmul, in exact integers.

FLOPs are ``2 * m * n * k``: one multiply and one add for each term of each
dot product. Two declared models count the bytes that move between global
memory and the SMs, for float32 elements of 4 bytes:

* ``no_reuse``: each thread that the guard admits loads its k elements of A
  and its k elements of B and stores its one element of C. No thread shares
  a load, and no cache serves one. The memory_access lab's
  ``NaiveMatmulTraffic`` counts the same elements.
* ``ideal_reuse``: each element of A and of B moves once, and each element of
  C is stored once, as if an on-chip store held all three matrices. This is
  the compulsory traffic of ``roofline.matmul_workload``.

The arithmetic intensity of a model is its FLOPs per byte, held as an exact
fraction. Without reuse it is ``k / (4 * k + 2)``, which stays below 1/4 for
every shape. No kernel moves fewer bytes than the compulsory traffic, so the
ideal-reuse intensity bounds every kernel of a shape from above. Both models
count requested elements, not sectors: the access model shows that a strided
warp moves a 32-byte sector for each 4-byte element, so the sector traffic of
a poor mapping can exceed the no-reuse bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .matmul_model_reference import require_shape
from .memory_access_tiling import MatmulShape, NaiveMatmulTraffic
from .roofline import matmul_workload

ELEMENT_BYTES = 4
TRAFFIC_MODELS = ("no_reuse", "ideal_reuse")
NO_REUSE_INTENSITY_LIMIT = Fraction(1, 4)
DECIMAL_PLACES = 6


def decimal_text(value: Fraction, places: int = DECIMAL_PLACES) -> str:
    """Write a fraction with ``places`` decimals, rounded half to even."""

    if type(value) is not Fraction:
        raise TypeError("value must be a Fraction")
    scaled = round(value * 10**places)
    whole, part = divmod(abs(scaled), 10**places)
    return f"{'-' if scaled < 0 else ''}{whole}.{part:0{places}d}"


def ratio_mapping(value: Fraction) -> dict[str, object]:
    """Return an exact ratio and a rounded decimal, so a report needs no float."""

    if type(value) is not Fraction:
        raise TypeError("value must be a Fraction")
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": decimal_text(value),
    }


@dataclass(frozen=True)
class MatmulTraffic:
    """The elements and bytes that one traffic model moves for one shape."""

    model: str
    shape: MatmulShape
    a_elements: int
    b_elements: int
    c_elements: int

    @property
    def flops(self) -> int:
        return self.shape.flops

    @property
    def bytes(self) -> int:
        return (self.a_elements + self.b_elements + self.c_elements) * ELEMENT_BYTES

    @property
    def intensity(self) -> Fraction:
        """Return FLOPs per byte of this model's traffic."""

        return Fraction(self.flops, self.bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "a_elements": self.a_elements,
            "b_elements": self.b_elements,
            "c_elements": self.c_elements,
            "bytes": self.bytes,
            "intensity": ratio_mapping(self.intensity),
        }


def no_reuse_traffic(shape: MatmulShape) -> MatmulTraffic:
    """Count k loads of A and of B and one store for each element of C."""

    checked = require_shape(shape)
    loads = checked.m * checked.n * checked.k
    return MatmulTraffic(
        TRAFFIC_MODELS[0], checked, loads, loads, checked.m * checked.n
    )


def ideal_reuse_traffic(shape: MatmulShape) -> MatmulTraffic:
    """Count each element of A, B and C once: the compulsory traffic."""

    checked = require_shape(shape)
    return MatmulTraffic(
        TRAFFIC_MODELS[1],
        checked,
        checked.m * checked.k,
        checked.k * checked.n,
        checked.m * checked.n,
    )


@dataclass(frozen=True)
class MatmulAccounting:
    """FLOPs and both traffic models of one shape, with their cross-checks."""

    shape: MatmulShape
    no_reuse: MatmulTraffic
    ideal_reuse: MatmulTraffic

    @property
    def flops(self) -> int:
        return self.shape.flops

    @property
    def reuse_factor(self) -> Fraction:
        """Return how many times the no-reuse bytes exceed the compulsory bytes."""

        return Fraction(self.no_reuse.bytes, self.ideal_reuse.bytes)

    def cross_checks(self) -> dict[str, bool]:
        """Compare both models with the memory_access and roofline modules."""

        m, n, k = self.shape.m, self.shape.n, self.shape.k
        naive = NaiveMatmulTraffic(self.shape, "float32")
        workload = matmul_workload(rows=m, inner=k, columns=n, dtype="float32")
        no_reuse, ideal = self.no_reuse, self.ideal_reuse
        return {
            "memory_access_naive_traffic": (
                naive.flops == self.flops
                and naive.load_elements == no_reuse.a_elements + no_reuse.b_elements
                and naive.store_elements == no_reuse.c_elements
                and naive.total_bytes == no_reuse.bytes
            ),
            "roofline_matmul_workload": (
                workload.flops == self.flops
                and workload.bytes_read
                == (ideal.a_elements + ideal.b_elements) * ELEMENT_BYTES
                and workload.bytes_written == ideal.c_elements * ELEMENT_BYTES
                and workload.total_bytes == ideal.bytes
            ),
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "shape": self.shape.to_mapping(),
            "flops": self.flops,
            "no_reuse": self.no_reuse.to_mapping(),
            "ideal_reuse": self.ideal_reuse.to_mapping(),
            "no_reuse_intensity_limit": ratio_mapping(NO_REUSE_INTENSITY_LIMIT),
            "reuse_factor": ratio_mapping(self.reuse_factor),
            "cross_checks": self.cross_checks(),
        }


def account_matmul(shape: MatmulShape) -> MatmulAccounting:
    """Return the FLOPs and both traffic models of one shape."""

    return MatmulAccounting(
        shape=require_shape(shape),
        no_reuse=no_reuse_traffic(shape),
        ideal_reuse=ideal_reuse_traffic(shape),
    )
