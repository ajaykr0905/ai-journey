"""Correctness of every Day 80 kernel on seeded cases, with determinism measured.

The run first measures the math probes: ``precise::exp`` must meet the Day
79 probe rule, and ``precise::rsqrt`` and ``precise::log`` must meet their
declared rules, or the run stops, because every bound assumes them. Each
case then draws seeded rows in a gapped layout, computes the float64
reference, and runs its family's kernel ``repeats`` times on the MPS
device, each run into a fresh buffer of sentinels.

Three rules apply to the output of every repeat, and a case that breaks one
stops the run, so a report never holds a failed check:

- Every gap slot and every slot after the last loss keeps its sentinel bits.
- Every output value is finite.
- Every value is within its declared bound of the float64 reference.

A race or a missed barrier can spoil one repeat only, so the rules check
each repeat. Two facts are measurements, not rules: ``identical_repeats``
records whether every repeat gave the bits of the first, and
``differing_from_rounded_reference`` counts the values whose bits differ
from the reference rounded to float32.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .kernel_metal_geometry import vector_launch
from .metal_fused_correctness import ProbeRecord
from .metal_fused_reference import (
    SENTINEL_BITS,
    exp_probe_arguments,
    exp_probe_errors,
    float32_sha256,
    float64_sha256,
)
from .metal_liger_config import LigerCase
from .metal_liger_reference import (
    CORRECTNESS_STREAM,
    MAX_CASE_INDEX,
    liger_inputs,
    liger_reference,
    log_probe_arguments,
    math_probe_errors,
    naive_cross_entropy_finite,
    rsqrt_probe_arguments,
)
from .metal_liger_source import CROSS_ENTROPY, MetalLigerError, require_size

RULES = (
    "every gap slot keeps its sentinel in every repeat; every output is finite; "
    "every value is within its bound of the reference; identical_repeats and "
    "differing_from_rounded_reference are measured"
)


def check_probes(backend: object) -> dict[str, dict[str, object]]:
    """Run the three probes once; stop when one breaks the assumption of a bound."""

    results = backend.probe("exp", exp_probe_arguments())
    exp = exp_probe_errors(results)
    if not exp.holds:
        raise MetalLigerError(
            "the exp probe breaks the assumption of the bounds: max relative "
            f"error {exp.max_relative_error!r}"
        )
    records = {
        "exp": ProbeRecord(float32_sha256(results), exp.to_mapping()).to_mapping()
    }
    for function, arguments in (
        ("rsqrt", rsqrt_probe_arguments()),
        ("log", log_probe_arguments()),
    ):
        probe = math_probe_errors(function, backend.probe(function, arguments))
        if not probe.holds:
            raise MetalLigerError(
                f"the {function} probe breaks the assumption of the bounds: max "
                f"relative error {probe.max_relative_error!r}"
            )
        records[function] = probe.to_mapping()
    return records


@dataclass(frozen=True)
class LigerOutput:
    """What one kernel's output showed in one case."""

    buffer_sha256: str
    identical_repeats: bool
    max_abs_error: float
    max_bound_ratio: float
    differing_from_rounded_reference: int

    def to_mapping(self) -> dict[str, object]:
        """Return the output as a report records it."""

        return {
            "buffer_sha256": self.buffer_sha256,
            "identical_repeats": self.identical_repeats,
            "max_abs_error": self.max_abs_error,
            "max_bound_ratio": self.max_bound_ratio,
            "differing_from_rounded_reference": self.differing_from_rounded_reference,
        }


@dataclass(frozen=True)
class LigerCaseRecord:
    """One correctness case: its inputs, reference, bound and kernel output."""

    index: int
    case: LigerCase
    repeats: int
    digests: dict[str, str]
    reference_sha256: str
    bound_sha256: str
    naive_float32_finite: bool | None
    output: LigerOutput

    def to_mapping(self) -> dict[str, object]:
        """Return the case with its digests, launch and output."""

        layout = self.case.layout()
        size = layout.group_size(self.case.family)
        return {
            "family": self.case.family,
            "index": self.index,
            "case": self.case.to_mapping(),
            "layout": layout.to_mapping(self.case.family),
            "launch": vector_launch(layout.rows * size, size).to_mapping(),
            "repeats": self.repeats,
            **self.digests,
            "reference_sha256": self.reference_sha256,
            "bound_sha256": self.bound_sha256,
            "naive_float32_finite": self.naive_float32_finite,
            "output": self.output.to_mapping(),
        }


def _buffer(result: object, size: int, where: str) -> numpy.ndarray:
    if (
        not isinstance(result, numpy.ndarray)
        or result.dtype != numpy.float32
        or result.shape != (size,)
    ):
        raise MetalLigerError(f"{where} returned no float32 buffer of its layout")
    return numpy.ascontiguousarray(result)


def check_case(
    backend: object,
    *,
    seed: int,
    index: int,
    case: LigerCase,
    repeats: int,
    required: int,
) -> LigerCaseRecord:
    """Run one case ``repeats`` times on its family's kernel, under the rules."""

    require_size(index, "case index", 0, MAX_CASE_INDEX)
    if not isinstance(case, LigerCase):
        raise TypeError("case must be a LigerCase")
    family, layout = case.family, case.layout()
    inputs = liger_inputs(seed, family, CORRECTNESS_STREAM, index, layout, case.inputs)
    reference = liger_reference(inputs)
    size = layout.output_elements(family)
    valid = layout.output_valid(family)
    gaps = numpy.ones(size, dtype=bool)
    gaps[valid] = False
    label = f"{family} case {index} ({layout.rows} rows, longest {layout.longest})"
    runs = []
    for number in range(1, repeats + 1):
        where = f"{label}: repeat {number} of {repeats}"
        run = _buffer(backend.run(inputs, required), size, where)
        if numpy.any(run[gaps].view(numpy.uint32) != SENTINEL_BITS):
            raise MetalLigerError(f"{where}: a store escaped into a gap slot")
        packed = run[valid]
        if not numpy.all(numpy.isfinite(packed)):
            raise MetalLigerError(f"{where}: an output is not finite")
        error = numpy.abs(packed.astype(numpy.float64) - reference.output)
        ratio = float(numpy.max(error / reference.bound))
        if not ratio <= 1.0:
            raise MetalLigerError(f"{where}: a value exceeds its bound by {ratio!r}")
        runs.append((run, error, ratio))
    first, error, ratio = runs[0]
    bits = first.view(numpy.uint32)
    rounded = reference.output.astype(numpy.float32).view(numpy.uint32)
    return LigerCaseRecord(
        index=index,
        case=case,
        repeats=repeats,
        digests=inputs.digests(),
        reference_sha256=float64_sha256(reference.output),
        bound_sha256=float64_sha256(reference.bound),
        naive_float32_finite=naive_cross_entropy_finite(inputs)
        if family == CROSS_ENTROPY
        else None,
        output=LigerOutput(
            buffer_sha256=float32_sha256(first),
            identical_repeats=all(
                numpy.array_equal(run.view(numpy.uint32), bits)
                for run, _, _ in runs[1:]
            ),
            max_abs_error=float(error.max()),
            max_bound_ratio=ratio,
            differing_from_rounded_reference=int(
                numpy.count_nonzero(bits[valid] != rounded)
            ),
        ),
    )
