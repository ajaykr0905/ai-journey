"""Correctness of every checklist kernel on seeded cases, with bit equality measured.

Each case draws seeded operands, computes the Day 74 float64 reference and
the CPU replay of the fma chain, and runs every kernel that its pipeline
admitted once on the MPS device: the two Day 74 kernels, the Day 76 tile-32
kernel and the Day 77 kernels. Each output buffer starts as sentinels and is
longer than C by every spare output slot of its launch, so a store that
escaped its guard would show in the tail.

Two rules apply to every output, and a case that breaks one stops the run,
so a report never holds a failed check:

- The guard tail must keep its sentinels.
- The largest relative error must stay within the Day 74 ``error_bound(K)``.

Bit equality with ``fma_replay`` is a measurement, not a rule. Each record
counts the output elements whose bits differ from the replay. Every
checklist kernel runs the replay's chain, one fma per k in increasing
order, so equal bits are possible. A host can still differ, and nothing
downstream assumes equality. The records reuse the Day 76 case and output
types, so both labs store a case in one form.
"""

from __future__ import annotations

import numpy

from .kernel_metal_geometry import require_integer
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS
from .matmul_checklist_runtime import checked_launch
from .matmul_checklist_source import MatmulChecklistError
from .matmul_checklist_variants import VARIANTS, guard_elements
from .matmul_metal_mapping import matmul_shape
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
from .matmul_tiled_correctness import KernelOutput, TiledCase
from .memory_access_tiling import MatmulShape


def _buffer(result: object, length: int, label: str) -> numpy.ndarray:
    if (
        not isinstance(result, numpy.ndarray)
        or result.dtype != numpy.float32
        or result.shape != (length,)
    ):
        raise MatmulChecklistError(f"{label} returned no float32 buffer of {length}")
    return numpy.ascontiguousarray(result)


def runnable(backend: object, names: object) -> tuple[str, ...]:
    """Return ``names`` once each one may launch; a refused kernel is an error.

    A refusal is already a record of the run, so asking to launch the kernel
    is a mistake of the caller.
    """

    chosen = tuple(names)
    for name in chosen:
        if name not in VARIANTS:
            raise MatmulChecklistError(f"{name!r} is not a registered variant")
        admission = backend.admissions.get(name)
        if admission is not None and not admission.admitted:
            raise MatmulChecklistError(
                f"{name} was refused by its pipeline and never launches"
            )
    return chosen


def check_case(
    backend: object,
    *,
    seed: int,
    index: int,
    shape: MatmulShape,
    naive_group_size: tuple[int, int],
    names: tuple[str, ...],
) -> TiledCase:
    """Run one case on every named kernel, in order, and enforce both rules."""

    require_integer(index, "case index", 0, MAX_CASE_INDEX)
    if not isinstance(shape, MatmulShape):
        raise TypeError("shape must be a MatmulShape")
    names = runnable(backend, names)
    shape = matmul_shape(shape.m, shape.n, shape.k)
    left, right = matmul_inputs(seed, CORRECTNESS_STREAM, index, shape)
    reference, magnitude = float64_reference(left, right)
    replay = fma_replay(left, right)
    replay_bits = replay.reshape(-1).view(numpy.uint32)
    bound = error_bound(shape.k)
    label = f"case {index} ({shape.m} x {shape.n} x {shape.k})"
    elements = shape.m * shape.n
    launches, outputs = [], []
    for name in names:
        launch = checked_launch(backend.limits, name, shape, naive_group_size)
        guard = guard_elements(VARIANTS[name], shape, launch.geometry)
        where = f"{label}: {name}"
        buffer = _buffer(
            backend.run(name, left, right, launch), elements + guard, where
        )
        if numpy.any(buffer[elements:].view(numpy.uint32) != FLOAT32_SENTINEL_BITS):
            raise MatmulChecklistError(f"{where}: a store escaped into the guard tail")
        output = buffer[:elements]
        error = max_relative_error(
            output.reshape(shape.m, shape.n), reference, magnitude
        )
        if not error <= bound:
            raise MatmulChecklistError(
                f"{where} has a relative error of {error!r}; the bound is {bound!r}"
            )
        differing = numpy.count_nonzero(output.view(numpy.uint32) != replay_bits)
        launches.append(launch)
        outputs.append(
            KernelOutput(
                kernel=name,
                correctness_mode=VARIANTS[name].correctness_mode,
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
