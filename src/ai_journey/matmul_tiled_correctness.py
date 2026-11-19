"""Correctness of the tiled kernels on seeded cases, with bit equality measured.

Each case draws seeded operands, computes the Day 74 float64 reference and
the CPU replay of the fma chain, and runs the naive Day 74 kernel and every
admitted variant once on the MPS device. Each output buffer starts as
sentinels and is longer than C by its launch's padding threads, so a write
by a padding thread would show in the tail.

Two rules apply to every output, and a case that breaks one stops the run,
so a report never holds a failed check:

- The guard tail must keep its sentinels.
- The largest relative error must stay within the Day 74 ``error_bound(K)``.

Bit equality with ``fma_replay`` is a measurement, not a rule. Each record
counts the output elements whose bits differ from the replay, and its
``replay_bits`` is true only when that count is 0. A variant that declares
the ``k_order_fma`` mode can still differ on some host; its record then
shows how many elements differ, and nothing downstream assumes equality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .kernel_metal_geometry import CheckedLaunch, require_integer
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS
from .matmul_metal_mapping import X_COLUMNS, matmul_launch, matmul_shape
from .matmul_metal_reference import (
    CORRECTNESS_STREAM,
    MAX_CASE_INDEX,
    error_bound,
    float64_reference,
    fma_replay,
    matmul_inputs,
    matrix_sha256,
    max_relative_error,
)
from .matmul_tiled_runtime import NAIVE_KERNEL
from .matmul_tiled_source import MatmulTiledError
from .matmul_tiled_variants import K_ORDER_FMA, TiledVariant
from .memory_access_tiling import MatmulShape


@dataclass(frozen=True)
class KernelOutput:
    """One kernel's output in one case: its digest, its error and its bits."""

    kernel: str
    correctness_mode: str
    buffer_sha256: str
    guard_elements: int
    max_relative_error: float
    differing_elements: int

    @property
    def replay_bits(self) -> bool:
        """Whether every output element held the replay's bits, as measured."""

        return self.differing_elements == 0

    def to_mapping(self, bound: float) -> dict[str, object]:
        """Return the record as a report stores it, with its share of the bound."""

        return {
            "kernel": self.kernel,
            "correctness_mode": self.correctness_mode,
            "buffer_sha256": self.buffer_sha256,
            "guard_elements": self.guard_elements,
            "max_relative_error": self.max_relative_error,
            "bound_ratio": self.max_relative_error / bound,
            "differing_elements": self.differing_elements,
            "replay_bits": self.replay_bits,
        }


@dataclass(frozen=True)
class TiledCase:
    """One seeded case, checked on the naive kernel and every admitted variant."""

    index: int
    shape: MatmulShape
    launches: tuple[CheckedLaunch, ...]
    a_sha256: str
    b_sha256: str
    reference_sha256: str
    replay_sha256: str
    outputs: tuple[KernelOutput, ...]

    @property
    def bound(self) -> float:
        """The largest relative error this case's K allows."""

        return error_bound(self.shape.k)

    def to_mapping(self) -> dict[str, object]:
        """Return the case as a report stores it."""

        return {
            "index": self.index,
            **self.shape.to_mapping(),
            "launches": {
                output.kernel: launch.to_mapping()
                for output, launch in zip(self.outputs, self.launches)
            },
            "a_sha256": self.a_sha256,
            "b_sha256": self.b_sha256,
            "reference_sha256": self.reference_sha256,
            "replay_sha256": self.replay_sha256,
            "bound": self.bound,
            "outputs": [output.to_mapping(self.bound) for output in self.outputs],
        }


def _buffer(result: object, length: int, label: str) -> numpy.ndarray:
    if (
        not isinstance(result, numpy.ndarray)
        or result.dtype != numpy.float32
        or result.shape != (length,)
    ):
        raise MatmulTiledError(f"{label} returned no float32 buffer of {length}")
    return numpy.ascontiguousarray(result)


def check_case(
    backend: object,
    *,
    seed: int,
    index: int,
    shape: MatmulShape,
    naive_group_size: tuple[int, int],
    variants: tuple[TiledVariant, ...],
) -> TiledCase:
    """Run one case on the naive kernel and each variant; enforce both rules.

    ``variants`` must all be admitted by their pipelines; a refused one is
    an error here, because its refusal is already a record of the run.
    """

    require_integer(index, "case index", 0, MAX_CASE_INDEX)
    if not isinstance(shape, MatmulShape):
        raise TypeError("shape must be a MatmulShape")
    if type(variants) is not tuple or not all(
        isinstance(variant, TiledVariant) for variant in variants
    ):
        raise TypeError("variants must be a tuple of TiledVariant values")
    for variant in variants:
        if not backend.admissions[variant.name].admitted:
            raise MatmulTiledError(
                f"{variant.name} was refused by its pipeline and never launches"
            )
    shape = matmul_shape(shape.m, shape.n, shape.k)
    left, right = matmul_inputs(seed, CORRECTNESS_STREAM, index, shape)
    reference, magnitude = float64_reference(left, right)
    replay = fma_replay(left, right)
    replay_bits = replay.reshape(-1).view(numpy.uint32)
    bound = error_bound(shape.k)
    label = f"case {index} ({shape.m} x {shape.n} x {shape.k})"
    elements = shape.m * shape.n
    kernels = [(NAIVE_KERNEL, naive_group_size, K_ORDER_FMA)]
    kernels += [
        (item.name, item.group_size, item.correctness_mode) for item in variants
    ]
    launches, outputs = [], []
    for name, group, mode in kernels:
        launch = CheckedLaunch(
            matmul_launch(X_COLUMNS, shape, group), backend.limits[name]
        )
        guard = launch.geometry.padding_threads
        where = f"{label}: {name}"
        buffer = _buffer(
            backend.run(name, left, right, launch), elements + guard, where
        )
        if numpy.any(buffer[elements:].view(numpy.uint32) != FLOAT32_SENTINEL_BITS):
            raise MatmulTiledError(
                f"{where}: a padding thread wrote into the guard tail"
            )
        output = buffer[:elements]
        error = max_relative_error(
            output.reshape(shape.m, shape.n), reference, magnitude
        )
        if not error <= bound:
            raise MatmulTiledError(
                f"{where} has a relative error of {error!r}; the bound is {bound!r}"
            )
        differing = numpy.count_nonzero(output.view(numpy.uint32) != replay_bits)
        launches.append(launch)
        outputs.append(
            KernelOutput(
                kernel=name,
                correctness_mode=mode,
                buffer_sha256=matrix_sha256(buffer),
                guard_elements=guard,
                max_relative_error=error,
                differing_elements=int(differing),
            )
        )
    return TiledCase(
        index=index,
        shape=shape,
        launches=tuple(launches),
        a_sha256=matrix_sha256(left),
        b_sha256=matrix_sha256(right),
        reference_sha256=matrix_sha256(reference),
        replay_sha256=matrix_sha256(replay),
        outputs=tuple(outputs),
    )
