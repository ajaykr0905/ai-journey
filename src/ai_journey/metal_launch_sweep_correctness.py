"""Check that every admitted shape computes the same correct result.

A matmul case draws standard-normal float32 ``A[m, k]`` and ``B[k, n]`` from
``numpy.random.default_rng([seed, MATMUL_STREAM, index])`` and runs the kernel
once with each admitted shape. The output buffer is longer than ``C`` by the
padding threads of the launch and starts as sentinels. Three rules apply:

1. The guard tail of each buffer keeps its sentinels.
2. Every shape gives the same ``C`` bit for bit, because the shape must not
   change the arithmetic of a thread.
3. Each element lies within ``(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @
   |B|)`` of the float64 reference, where ``gamma_K(u) = K u / (1 - K u)``
   (N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.,
   SIAM, 2002, section 3.1). The bound holds for every summation order and
   for fused multiply-adds, so a correct kernel cannot fail it.

``memory_access_baseline`` applies the same bound to CPU matmuls, but that
module imports torch when it loads and stops at ``K = 1024``, so this module
states the bound for its own sizes. A record keeps the digest of ``C``, the
largest error and bound ratio, and the values of ``C`` at a lattice of
positions. A verifier can thus redraw the inputs and recheck those values
against its own float64 reference.

The 1D sweep reuses the Day 73 ``check_vector_add`` for each admitted group
size, at one length that is not a multiple of the group size: the Metal
kernel, PyTorch on MPS and PyTorch on the CPU must match the float32 sum bit
for bit, and the guard tail must keep its sentinels. A case that breaks a
rule stops the run, so a report never holds a failed check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .kernel_metal_correctness import VectorAddCase, check_vector_add
from .kernel_metal_geometry import require_integer
from .kernel_metal_reference import (
    FLOAT32_SENTINEL_BITS,
    MAX_CASE_INDEX,
    MAX_SEED,
    array_sha256,
)
from .metal_launch_sweep_shapes import (
    MAX_MATRIX_SIDE,
    MetalLaunchSweepError,
    ShapeAdmission,
    checked_launch,
    matmul_launch,
    shape_label,
)

# Day 73 drew vector add from stream 0 and grayscale from stream 1.
MATMUL_STREAM = 2
FLOAT32_UNIT_ROUNDOFF = 2.0**-24
FLOAT64_UNIT_ROUNDOFF = 2.0**-53
LATTICE_POINTS = 5
BOUND = "(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)"
REFERENCE = "float64 matmul on the CPU, rebuilt from the seed"


def _gamma(inner: int, unit_roundoff: float) -> float:
    product = inner * unit_roundoff
    return product / (1 - product)


def matmul_tolerance(k: int) -> float:
    """Return the relative bound of one float32 dot product of length ``k``."""

    inner = require_integer(k, "k", 1, MAX_MATRIX_SIDE)
    return _gamma(inner, FLOAT32_UNIT_ROUNDOFF) + _gamma(inner, FLOAT64_UNIT_ROUNDOFF)


def matmul_inputs(
    seed: int, index: int, m: int, n: int, k: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Draw the seeded standard-normal float32 operands of one matmul."""

    require_integer(seed, "seed", 0, MAX_SEED)
    require_integer(index, "matmul index", 0, MAX_CASE_INDEX)
    for name, size in (("m", m), ("n", n), ("k", k)):
        require_integer(size, name, 1, MAX_MATRIX_SIDE)
    generator = numpy.random.default_rng([seed, MATMUL_STREAM, index])
    a = generator.standard_normal((m, k), dtype=numpy.float32)
    return a, generator.standard_normal((k, n), dtype=numpy.float32)


def lattice(extent: int) -> tuple[int, ...]:
    """Return evenly spread positions from 0 to ``extent - 1``, both included."""

    last = require_integer(extent, "extent", 1, MAX_MATRIX_SIDE) - 1
    steps = LATTICE_POINTS - 1
    return tuple(sorted({point * last // steps for point in range(LATTICE_POINTS)}))


@dataclass(frozen=True)
class MatmulAccuracy:
    """The float64 comparison of one ``C`` and the values a verifier rechecks."""

    output_sha256: str
    max_abs_error: float
    max_bound_ratio: float
    samples: tuple[tuple[int, int, float], ...]

    def to_mapping(self) -> dict[str, object]:
        """Return the comparison as a report records it."""

        return {
            "output_sha256": self.output_sha256,
            "max_abs_error": self.max_abs_error,
            "max_bound_ratio": self.max_bound_ratio,
            "samples": [list(sample) for sample in self.samples],
        }


def matmul_accuracy(
    a: numpy.ndarray, b: numpy.ndarray, c: numpy.ndarray, label: str
) -> MatmulAccuracy:
    """Hold every element of ``C`` to the float32 bound of the float64 product."""

    m, k = a.shape
    n = b.shape[1]
    if not isinstance(c, numpy.ndarray) or c.dtype != numpy.float32 or c.size != m * n:
        raise MetalLaunchSweepError(f"{label}: the kernel returned no {m}x{n} result")
    output = c.reshape(m, n)
    if not numpy.isfinite(output).all():
        raise MetalLaunchSweepError(f"{label}: the result is not finite")
    left, right = a.astype(numpy.float64), b.astype(numpy.float64)
    error = numpy.abs(output.astype(numpy.float64) - left @ right)
    bound = matmul_tolerance(k) * (numpy.abs(left) @ numpy.abs(right))
    if bool((error > bound).any()):
        raise MetalLaunchSweepError(
            f"{label}: the result exceeds the float32 rounding bound {BOUND}"
        )
    positive = bound > 0
    ratio = float((error[positive] / bound[positive]).max(initial=0.0))
    samples = tuple(
        (row, column, float(output[row, column]))
        for row in lattice(m)
        for column in lattice(n)
    )
    return MatmulAccuracy(
        output_sha256=array_sha256(output),
        max_abs_error=float(error.max()),
        max_bound_ratio=ratio,
        samples=samples,
    )


@dataclass(frozen=True)
class MatmulCase:
    """One seeded matmul that every admitted shape computed identically."""

    index: int
    m: int
    n: int
    k: int
    a_sha256: str
    b_sha256: str
    guard_elements: tuple[tuple[tuple[int, int], int], ...]
    accuracy: MatmulAccuracy

    def to_mapping(self) -> dict[str, object]:
        """Return the case as a report records it."""

        return {
            "index": self.index,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "a_sha256": self.a_sha256,
            "b_sha256": self.b_sha256,
            "shapes": [
                {"group_size": list(group), "guard_elements": guard}
                for group, guard in self.guard_elements
            ],
            **self.accuracy.to_mapping(),
        }


def _admitted(admissions: tuple[ShapeAdmission, ...]) -> tuple[ShapeAdmission, ...]:
    chosen = tuple(item for item in admissions if item.admitted)
    if not chosen:
        raise MetalLaunchSweepError("no shape of the sweep was admitted")
    return chosen


def check_matmul_case(
    backend: object,
    *,
    seed: int,
    index: int,
    shape: tuple[int, int, int],
    admissions: tuple[ShapeAdmission, ...],
) -> MatmulCase:
    """Run one case with every admitted shape and require one exact answer."""

    m, n, k = shape
    a, b = matmul_inputs(seed, index, m, n, k)
    label = f"matmul case {index} ({m}x{n}x{k})"
    first: numpy.ndarray | None = None
    guards = []
    for admission in _admitted(admissions):
        launch = checked_launch(admission, matmul_launch(m, n, admission.group_size))
        guard = launch.geometry.padding_threads
        buffer = backend.matmul(a, b, launch)
        name = f"{label} with group {shape_label(admission.group_size)}"
        if (
            not isinstance(buffer, numpy.ndarray)
            or buffer.dtype != numpy.float32
            or buffer.shape != (m * n + guard,)
        ):
            raise MetalLaunchSweepError(f"{name}: no {m * n + guard}-element buffer")
        if numpy.any(buffer[m * n :].view(numpy.uint32) != FLOAT32_SENTINEL_BITS):
            raise MetalLaunchSweepError(f"{name}: a padding thread wrote the tail")
        output = buffer[: m * n]
        if first is None:
            first = output
        elif not numpy.array_equal(output.view(numpy.uint32), first.view("u4")):
            raise MetalLaunchSweepError(f"{name}: the result differs between shapes")
        guards.append((admission.group_size, guard))
    return MatmulCase(
        index=index,
        m=m,
        n=n,
        k=k,
        a_sha256=array_sha256(a),
        b_sha256=array_sha256(b),
        guard_elements=tuple(guards),
        accuracy=matmul_accuracy(a, b, first, label),
    )


def check_vector_sizes(
    backend: object,
    *,
    seed: int,
    length: int,
    admissions: tuple[ShapeAdmission, ...],
) -> tuple[VectorAddCase, ...]:
    """Run the Day 73 vector-add check once per admitted group size.

    Each size keeps its position in the sweep as its case index, so a
    rejected size leaves a gap instead of renumbering the others.
    """

    _admitted(admissions)
    return tuple(
        check_vector_add(
            backend,
            seed=seed,
            index=position,
            length=length,
            group_size=admission.group_size[0],
        )
        for position, admission in enumerate(admissions)
        if admission.admitted
    )
