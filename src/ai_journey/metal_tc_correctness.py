"""Correctness of the four kernels on seeded ragged cases, with determinism measured.

Each case draws seeded float32 operands, computes the float64 reference of
every output and runs each supported kernel ``repeats`` times on the MPS
device, each run into a fresh C buffer that ends in sentinels. Every
operand ends in quiet NaNs, so a read past an operand poisons an output.

Three rules apply to the output of every repeat, and a case that breaks one
stops the run, so a report never holds a failed check:

- Every tail element of the C buffer keeps its sentinel bits.
- Every output is finite.
- Every output is within the bound of its input type.

A race or a missed barrier can spoil one repeat only, so the rules check
each repeat. Two facts are measurements, not rules: ``identical_repeats``
records whether every repeat gave the bits of the first, and
``m1_same_bits_as_s1`` records whether the matrix path gave the bits of
the scalar fma chain on the same float32 operands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .metal_flash_reference import float64_digest
from .metal_fused_reference import float32_sha256
from .metal_tc_config import TcCase
from .metal_tc_reference import (
    CORRECTNESS_STREAM,
    MAX_CASE_INDEX,
    SENTINEL_BITS,
    TAIL,
    TcReference,
    operands,
    tc_reference,
)
from .metal_tc_runtime import launch_mapping
from .metal_tc_source import (
    M1,
    PRECISION,
    PRECISIONS,
    S1,
    VARIANTS,
    MetalTcError,
    require_size,
)

RULES = (
    "every C buffer keeps its tail sentinels in every repeat; every output is "
    "finite; every output is within the bound of its input type; "
    "identical_repeats and m1_same_bits_as_s1 are measured"
)


@dataclass(frozen=True)
class TcOutput:
    """What one kernel's output showed in one case."""

    buffer_sha256: str
    identical_repeats: bool
    max_abs_error: float
    max_bound_ratio: float
    max_relative_error: float

    def to_mapping(self) -> dict[str, object]:
        """Return the output as a report records it."""

        return {
            "buffer_sha256": self.buffer_sha256,
            "identical_repeats": self.identical_repeats,
            "max_abs_error": self.max_abs_error,
            "max_bound_ratio": self.max_bound_ratio,
            "max_relative_error": self.max_relative_error,
        }


def compare(
    output: numpy.ndarray, reference: TcReference, bound: numpy.ndarray, where: str
) -> tuple[float, float, float]:
    """Check the rows of one flat C; return the largest error, ratio and share of S."""

    columns = reference.output.shape[1]
    rows = output.reshape(-1, columns)[reference.rows]
    if not numpy.all(numpy.isfinite(rows)):
        raise MetalTcError(f"{where}: an output is not finite")
    error = numpy.abs(rows.astype(numpy.float64) - reference.output)
    ratio = float(numpy.max(error / bound))
    if not ratio <= 1.0:
        raise MetalTcError(f"{where}: an output exceeds its bound by {ratio!r}")
    relative = float(numpy.max(error / reference.magnitude))
    return float(error.max()), ratio, relative


def _buffer(result: object, size: int, where: str) -> numpy.ndarray:
    if (
        not isinstance(result, numpy.ndarray)
        or result.dtype != numpy.float32
        or result.shape != (size + TAIL,)
    ):
        raise MetalTcError(f"{where} returned no float32 C buffer with its tail")
    return numpy.ascontiguousarray(result)


def check_variant(
    backend: object,
    values: object,
    reference: TcReference,
    variant: str,
    repeats: int,
    label: str,
) -> tuple[TcOutput, numpy.ndarray]:
    """Run one kernel ``repeats`` times on one case, under the rules."""

    size = values.shape.m * values.shape.n
    bound = reference.bound(PRECISION[variant])
    runs = []
    for number in range(1, repeats + 1):
        where = f"{label} {variant}: repeat {number} of {repeats}"
        run = _buffer(backend.run(values, variant), size, where)
        if numpy.any(run[size:].view(numpy.uint32) != SENTINEL_BITS):
            raise MetalTcError(f"{where}: a store escaped into the tail")
        runs.append((run, *compare(run[:size], reference, bound, where)))
    first, error, ratio, relative = runs[0]
    bits = first.view(numpy.uint32)
    record = TcOutput(
        buffer_sha256=float32_sha256(first),
        identical_repeats=all(
            numpy.array_equal(run.view(numpy.uint32), bits) for run, *_ in runs[1:]
        ),
        max_abs_error=error,
        max_bound_ratio=ratio,
        max_relative_error=relative,
    )
    return record, bits


def bound_digests(reference: TcReference) -> dict[str, str]:
    """The digest of the bound of every output for each input type."""

    return {
        precision: float64_digest(reference.bound(precision))
        for precision in PRECISIONS
    }


def check_case(
    backend: object, *, seed: int, index: int, case: TcCase, repeats: int
) -> dict[str, object]:
    """Run one case on every supported kernel ``repeats`` times, under the rules."""

    require_size(index, "case index", 0, MAX_CASE_INDEX)
    if not isinstance(case, TcCase):
        raise TypeError("case must be a TcCase")
    shape = case.shape()
    values = operands(seed, CORRECTNESS_STREAM, index, shape)
    reference = tc_reference(values)
    label = f"case {index} ({case.m} x {case.n} x {case.k})"
    outputs, bits = {}, {}
    for variant in VARIANTS:
        if not backend.supported(variant):
            continue
        record, bits[variant] = check_variant(
            backend, values, reference, variant, repeats, label
        )
        outputs[variant] = record.to_mapping()
    return {
        "index": index,
        "case": case.to_mapping(),
        "repeats": repeats,
        "launch": launch_mapping(shape),
        "inputs": values.digests(),
        **reference.digests(),
        "bound_sha256": bound_digests(reference),
        "outputs": outputs,
        "m1_same_bits_as_s1": bool(numpy.array_equal(bits[M1], bits[S1])),
    }
