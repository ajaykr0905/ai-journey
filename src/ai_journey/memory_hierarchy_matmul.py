"""A checked NumPy matmul whose measured FLOP rate is the CPU compute ceiling.

``numpy.matmul(left, right, product)`` multiplies seeded standard-normal
operands of the configured dtype into a preallocated output, and each timed
sample is exactly one call. FLOPs are ``2 * M * N * K``, one multiply and one
add per scalar product term, taken from ``roofline.matmul_workload`` together
with its compulsory read-once, write-once traffic. NumPy hands the call to
its BLAS library, which may run several threads, while the STREAM kernels
and gathers run on one thread; the compute ceiling and the bandwidths need
not share a thread count.

After the clock stops, every element of the product is checked against a
float64 reference, one block of rows at a time. Each must lie within
``(gamma_K(u) + gamma_K(2**-53)) * (|A| @ |B|)``, where ``u`` is the dtype's
unit roundoff and ``gamma_K(u) = K u / (1 - K u)`` is the classical
inner-product rounding bound (N. J. Higham, Accuracy and Stability of
Numerical Algorithms, 2nd ed., SIAM, 2002, section 3.1). The bound holds for
any summation order, so a correct BLAS cannot fail it while a wrong product
does.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy

from .memory_hierarchy_stream import (
    MAX_SEED,
    MemoryHierarchyError,
    element_bytes,
    require_integer,
)
from .memory_hierarchy_timing import (
    BatchTiming,
    RunBudget,
    per_nanosecond,
    time_batches,
)
from .roofline import KernelWorkload, matmul_workload

MIN_MATMUL_DIMENSION = 16
MAX_MATMUL_DIMENSION = 2048
CHECK_ROW_BLOCK = 64
UNIT_ROUNDOFF = {"float32": 2.0**-24, "float64": 2.0**-53}
CORRECTNESS_BOUND = "(gamma_K(u) + gamma_K(2**-53)) * (|A| @ |B|)"


def validate_matmul(m: object, n: object, k: object) -> tuple[int, int, int]:
    """Bound the matmul so its operands, checks and FLOP counts stay small."""

    return tuple(
        require_integer(value, name, MIN_MATMUL_DIMENSION, MAX_MATMUL_DIMENSION)
        for name, value in (("m", m), ("n", n), ("k", k))
    )


def rounding_tolerance(inner: int, dtype: str) -> float:
    """Return the relative bound one checked product element must satisfy.

    It covers the BLAS result's own rounding plus the float64 reference's.
    """

    count = require_integer(inner, "k", 1, MAX_MATMUL_DIMENSION)
    element_bytes(dtype)
    total = 0.0
    for unit in (UNIT_ROUNDOFF[dtype], UNIT_ROUNDOFF["float64"]):
        product = count * unit
        total += product / (1 - product)
    return total


def check_matmul(
    left: numpy.ndarray, right: numpy.ndarray, product: numpy.ndarray
) -> float:
    """Check every product element against float64; return the worst bound ratio.

    The ratio is the largest ``|error| / bound`` seen; anything above one
    fails, so a recorded ratio shows how much of the bound was used. Row
    blocks keep the float64 temporaries small next to the operands.
    """

    if (
        not all(isinstance(item, numpy.ndarray) for item in (left, right, product))
        or left.ndim != 2
        or right.ndim != 2
        or product.shape != (left.shape[0], right.shape[1])
        or left.shape[1] != right.shape[0]
        or not left.dtype == right.dtype == product.dtype
    ):
        raise MemoryHierarchyError("matmul operands and product are not conformable")
    tolerance = rounding_tolerance(left.shape[1], str(left.dtype))
    right64 = right.astype(numpy.float64)
    magnitude = numpy.abs(right64)
    worst = 0.0
    for start in range(0, left.shape[0], CHECK_ROW_BLOCK):
        rows = slice(start, start + CHECK_ROW_BLOCK)
        observed = product[rows].astype(numpy.float64)
        if not numpy.isfinite(observed).all():
            raise MemoryHierarchyError("matmul product is not finite")
        left64 = left[rows].astype(numpy.float64)
        error = numpy.abs(observed - left64 @ right64)
        bound = tolerance * (numpy.abs(left64) @ magnitude)
        if bool((error > bound).any()):
            raise MemoryHierarchyError("matmul product exceeds its rounding bound")
        positive = bound > 0
        worst = max(worst, float((error[positive] / bound[positive]).max(initial=0.0)))
    return worst


def matmul_allocation_bytes(m: int, n: int, k: int, dtype: str) -> int:
    """Return the bytes a matmul measurement allocates, including its check.

    The operands and product, the check's float64 right operand and its
    absolute values, and one row block of float64 temporaries; the run's
    memory bound uses this.
    """

    rows, columns, inner = validate_matmul(m, n, k)
    width = element_bytes(dtype)
    operands = (rows * inner + inner * columns + rows * columns) * width
    block = min(rows, CHECK_ROW_BLOCK) * (2 * inner + 4 * columns) * 8
    return operands + 2 * inner * columns * 8 + block


@dataclass(frozen=True)
class MatmulMeasurement:
    """One checked matmul shape timed one call per sample."""

    m: int
    n: int
    k: int
    dtype: str
    timing: BatchTiming
    max_bound_ratio: float

    @property
    def workload(self) -> KernelWorkload:
        """FLOPs and compulsory bytes from ``roofline.matmul_workload``."""

        return matmul_workload(
            rows=self.m, inner=self.k, columns=self.n, dtype=self.dtype
        )

    @property
    def median_gflops_per_second(self) -> float:
        """FLOPs per nanosecond at the median sample (GFLOP/s)."""

        return per_nanosecond(self.workload.flops, self.timing.summary.median_ns)

    def to_mapping(self) -> dict[str, object]:
        """Return the raw samples, FLOP accounting, rates and check result."""

        workload = self.workload
        return {
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "dtype": self.dtype,
            "call": "numpy.matmul(left, right, out=product)",
            "flops_per_call": workload.flops,
            "compulsory_bytes_per_call": workload.total_bytes,
            **self.timing.to_mapping(),
            "median_gflops_per_second": self.median_gflops_per_second,
            "best_gflops_per_second": per_nanosecond(
                workload.flops, self.timing.summary.min_ns
            ),
            "correctness": {
                "reference": "float64",
                "bound": CORRECTNESS_BOUND,
                "unit_roundoff": UNIT_ROUNDOFF[self.dtype],
                "tolerance": rounding_tolerance(self.k, self.dtype),
                "checked_elements": self.m * self.n,
                "max_bound_ratio": self.max_bound_ratio,
            },
        }


def measure_matmul(
    *,
    m: int,
    n: int,
    k: int,
    dtype: str,
    seed: int,
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> MatmulMeasurement:
    """Time one call per sample of a seeded matmul, then check its product.

    The product starts as NaN, so a call that wrote nothing cannot pass.
    """

    rows, columns, inner = validate_matmul(m, n, k)
    element_bytes(dtype)
    kind = numpy.dtype(dtype)
    generator = numpy.random.default_rng(require_integer(seed, "seed", 0, MAX_SEED))
    left = generator.standard_normal((rows, inner), dtype=kind)
    right = generator.standard_normal((inner, columns), dtype=kind)
    product = numpy.full((rows, columns), numpy.nan, dtype=kind)
    timing = time_batches(
        partial(numpy.matmul, left, right, product),
        calls_per_sample=1,
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    return MatmulMeasurement(
        m=rows,
        n=columns,
        k=inner,
        dtype=dtype,
        timing=timing,
        max_bound_ratio=check_matmul(left, right, product),
    )
